// Package mujocoenv implements environments that use the MuJoCo
// physics simulator
package mujocoenv

// * Leaving the cgo directives in so VSCode doesn't complain, even though
// * CGO_CFLAGS and CGO_LDFLAGS have been set.
// #cgo CFLAGS: -O2 -I/home/samuel/.mujoco/mujoco200_linux/include -mavx -pthread
// #cgo LDFLAGS: -L/home/samuel/.mujoco/mujoco200_linux/bin -lmujoco200nogl
// #include "mujoco.h"
// #include <stdio.h>
// #include <stdlib.h>
//
// void setQPos(mjData* data, double* positions, int len) {
// for (int i = 0; i < len; i++) {
// 		*((*data).qpos + i) = *(positions + i);
// 	}
// }
//
// void setQVel(mjData* data, double* velocities, int len) {
// 	for (int i = 0; i < len; i++){
// 		*((*data).qvel + i) = *(velocities + i);
// 	}
// }
//
// void setControl(mjData* data, double* control, int len) {
// 	for (int i = 0; i < len; i++){
// 		*((*data).ctrl + i) = *(control + i);
// 	}
// }
import "C"

import (
	"fmt"
	"os"
	"path"
	"unsafe"

	"github.com/samuelfneumann/golearn/environment"
	"gonum.org/v1/gonum/mat"
)

func init() {
	// Activate MuJoCo
	home, err := os.UserHomeDir()
	if err != nil {
		panic(fmt.Sprintf("could not find user home directory: %v", err))
	}
	mjKey := C.CString(fmt.Sprintf("%v/.mujoco/mjkey.txt", home))
	defer C.free(unsafe.Pointer(mjKey))
	C.mj_activate(mjKey)
}

type MujocoEnv struct {
	FrameSkip int
	Model     *C.mjModel
	Data      *C.mjData
	Seed      uint64
	Discount  float64

	InitQPos *mat.VecDense
	InitQVel *mat.VecDense

	Nu, Nv, Nq, Na int
}

func NewMujocoEnv(xmlPath string, frameSkip int, seed uint64,
	discount float64) (*MujocoEnv, error) {
	var fullPath string
	if xmlPath[0] == '/' || xmlPath[0:2] == "./" {
		fullPath = xmlPath
	} else {
		wd, err := os.Getwd()
		if err != nil {
			return nil, fmt.Errorf("newMujocoEnv: could not get current "+
				"directory for finding mujoco/assets/ dir: %v", err)
		}
		fullPath = path.Join(wd, "environment/mujoco/assets", xmlPath)
	}

	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("newMuJocoEnv: no such path '%v'", fullPath)
	}

	model, data, err := loadXML(fullPath)
	if err != nil {
		return nil, fmt.Errorf("newMujocoEnv: could not load XML: %v", err)
	}

	nq := int(model.nq)
	nu := int(model.nu)
	nv := int(model.nv)
	na := int(model.na)

	initQPos := F64SliceC2Go(data.qpos, nq)
	initQVel := F64SliceC2Go(data.qvel, nv)

	// Seed the environment
	C.srand(C.uint(seed))

	return &MujocoEnv{
		FrameSkip: frameSkip,
		Model:     model,
		Data:      data,
		Seed:      seed,
		Discount:  discount,
		Nu:        nu,
		Nv:        nv,
		Nq:        nq,
		Na:        na,
		InitQPos:  mat.NewVecDense(len(initQPos), initQPos),
		InitQVel:  mat.NewVecDense(len(initQVel), initQVel),
	}, nil
}

// "Concrete classes" should have a reset which calls this reset
func (m *MujocoEnv) Reset() {
	C.mj_resetData(m.Model, m.Data)
}

func (m *MujocoEnv) QPos() []float64 {
	return F64SliceC2Go(m.Data.qpos, m.Nq)
}

func (m *MujocoEnv) QVel() []float64 {
	return F64SliceC2Go(m.Data.qvel, m.Nq)
}

func (m *MujocoEnv) SetState(qpos, qvel []float64) error {
	if len(qpos) != m.Nq {
		return fmt.Errorf("setState: invalid position dimensions \n\t"+
			"have(%v) \n\twant(%v)", len(qpos), m.Nq)
	}
	if len(qvel) != m.Nv {
		return fmt.Errorf("setState: invalid velocity dimensions \n\t"+
			"\n\thave(%v) \n\twant(%v)", len(qvel), m.Nv)
	}

	// Set the state
	C.setQPos(m.Data, (*C.double)(unsafe.Pointer(&qpos[0])), C.int(len(qpos)))
	C.setQVel(m.Data, (*C.double)(unsafe.Pointer(&qvel[0])), C.int(len(qvel)))

	C.mj_forward(m.Model, m.Data)
	return nil
}

func (m *MujocoEnv) Dt() float64 {
	return float64(m.Model.opt.timestep) * float64(m.FrameSkip)
}

func (m *MujocoEnv) DoSimulation(control *mat.VecDense, nFrames int) error {
	if control.Len() != m.Nu {
		return fmt.Errorf("doSimulation: invalid control dimensions \n\t"+
			"have(%v) \n\twant(%v)", control.Len(), m.Nu)
	}

	action := make([]float64, control.Len())
	copy(action, control.RawVector().Data)
	C.setControl(m.Data, (*C.double)(unsafe.Pointer(&action[0])), C.int(len(action)))

	for i := 0; i < nFrames; i++ {
		C.mj_step(m.Model, m.Data)
	}
	return nil
}

func (m *MujocoEnv) DiscountSpec() environment.Spec {
	bounds := mat.NewVecDense(1, []float64{m.Discount})

	return environment.NewSpec(mat.NewVecDense(1, nil), environment.Discount,
		bounds, bounds, environment.Continuous)
}

func (m *MujocoEnv) ActionSpec() environment.Spec {
	bounds := F64SliceC2Go(m.Model.actuator_ctrlrange, m.Nu*2)

	low := make([]float64, m.Nu)
	high := make([]float64, m.Nu)
	for i := 0; i < m.Nu; i++ {
		low[i] = bounds[2*i]
		high[i] = bounds[2*i+1]
	}

	lowVec := mat.NewVecDense(m.Nu, low)
	highVec := mat.NewVecDense(m.Nu, high)
	shape := mat.NewVecDense(m.Nu, nil)

	return environment.NewSpec(shape, environment.Action, lowVec, highVec,
		environment.Continuous)
}

func (m *MujocoEnv) Close() {
	C.mj_deleteModel(m.Model)
	C.mj_deleteData(m.Data)
}
