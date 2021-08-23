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

// init performs setup before the package can be run by initialziing
// and activating MuJoCo
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

// MujocoEnv implements the base functionality for all environments
// using the MuJoCo simulator.
//
// MujocoEnv does not satisfy the environment.Environment interface.
//
// See http://www.mujoco.org/book/APIreference.html for more in-depth
// documentation.
type MujocoEnv struct {
	FrameSkip int // Number of frames the run per timestep

	Model *C.mjModel // MuJoCo model
	Data  *C.mjData  // MuJoCo data

	Seed     uint64
	Discount float64

	InitQPos *mat.VecDense // Initial position of the model
	InitQVel *mat.VecDense // Initial velocity of the model

	Nu int // Number of action dimensions
	Nv int // Number of velocity dimensions
	Nq int // Number of position dimensions
	Na int // Number of actuator activation  dimensions
}

// NewMujocoEnv returns a new MujocoEnv where the MuJoCo model is
// defined in the XML file at xmlPath. The frameSkip argument determines
// how many simulator frames should be run at each timestep.
func NewMujocoEnv(xmlPath string, frameSkip int, seed uint64,
	discount float64) (*MujocoEnv, error) {
	// Expand xmlPath to a full file path
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

	// Load the XML data to a MuJoCo model and data struct
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

// Reset resets the MuJoCo data and model
func (m *MujocoEnv) Reset() {
	C.mj_resetData(m.Model, m.Data)
}

// QPos returns the current position of the model
func (m *MujocoEnv) QPos() []float64 {
	return F64SliceC2Go(m.Data.qpos, m.Nq)
}

// QVel returns the current velocity of the model
func (m *MujocoEnv) QVel() []float64 {
	return F64SliceC2Go(m.Data.qvel, m.Nq)
}

// SetState sets the underlying position and velocity for the model
// in the environment.
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

// Dt returns dt, the difference in time between timesteps
func (m *MujocoEnv) Dt() float64 {
	return float64(m.Model.opt.timestep) * float64(m.FrameSkip)
}

// DoSimulation runs the simulator for nFrames given some control
func (m *MujocoEnv) DoSimulation(control *mat.VecDense, nFrames int) error {
	if control.Len() != m.Nu {
		return fmt.Errorf("doSimulation: invalid control dimensions \n\t"+
			"have(%v) \n\twant(%v)", control.Len(), m.Nu)
	}

	// Set the action
	// ? Copying the data can be removed, since setControl automatically
	// ? copies the data itself.
	action := make([]float64, control.Len())
	copy(action, control.RawVector().Data)
	C.setControl(m.Data, (*C.double)(unsafe.Pointer(&action[0])), C.int(len(action)))

	// Take nFrames simulator steps
	for i := 0; i < nFrames; i++ {
		C.mj_step(m.Model, m.Data)
	}
	return nil
}

// DiscountSpec returns the discount specification for the enviornment
func (m *MujocoEnv) DiscountSpec() environment.Spec {
	bounds := mat.NewVecDense(1, []float64{m.Discount})

	return environment.NewSpec(mat.NewVecDense(1, nil), environment.Discount,
		bounds, bounds, environment.Continuous)
}

// ActionSpec returns the action specification for the environment
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

// Close performs cleanup of MuJoCo resources once the environment
// is no longer needed.
func (m *MujocoEnv) Close() {
	C.mj_deleteModel(m.Model)
	C.mj_deleteData(m.Data)
}
