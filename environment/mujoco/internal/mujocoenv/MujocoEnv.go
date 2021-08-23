// Package mujocoenv implements environments that use the MuJoCo
// physics simulator
package mujocoenv

// * Leaving the cgo directives in so VSCode doesn't complain, even though
// * CGO_CFLAGS and CGO_LDFLAGS have been set.

// #cgo CFLAGS: -O2 -I/home/samuel/.mujoco/mujoco200_linux/include -mavx -pthread
// #cgo LDFLAGS: -L/home/samuel/.mujoco/mujoco200_linux/bin -lmujoco200nogl
// #define DIMS 3
// #include "mujoco.h"
// #include <stdio.h>
// #include <stdlib.h>
//
// // setQPos sets the position of the simulation body to position,
// // which has length len. If the returned value is less than 0,
// // an error occurred.
// int setQPos(mjData* data, mjModel* model, double* position, int len) {
// if (len != model->nq) {
// 	return -1;
// }
// for (int i = 0; i < len; i++) {
// 		*((*data).qpos + i) = *(position + i);
// 	}
//  return 1;
// }
//
// // setQVel sets the velocity of the simulation body to velocity,
// // which has length len. If the returned value is less than 0,
// // an error occurred.
// int setQVel(mjData* data, mjModel *model, double* velocity, int len) {
// if (len != model->nv) {
// 	return -1;
// }
// 	for (int i = 0; i < len; i++){
// 		*((*data).qvel + i) = *(velocity + i);
// 	}
// return 1;
// }
//
// // setControl sets the control of the simulation to control which
// // has length len. If the returned value is less than 0,
// // an error occurred.
// int setControl(mjData* data, mjModel *model, double* control, int len) {
// if (len != model->nu) {
// 	return -1;
// }
// 	for (int i = 0; i < len; i++){
// 		*((*data).ctrl + i) = *(control + i);
// 	}
// return 1;
// }
//
// // getCentreOfMass gets the centre of mass of the body with ID id
// // and places the centre of mass coordinates (x, y, z) in data.
// // mjBodyData is the mjData of the simulation.
// double* getCentreOfMass(int id, mjData *mjBodyData, double *data) {
//  for (int i = 0; i < DIMS; i++) {
// 		*(data + i) = *(mjBodyData->xpos + id + i);
//  }
// return data;
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
	errCode := C.setQPos(
		m.Data,
		m.Model,
		(*C.double)(unsafe.Pointer(&qpos[0])),
		C.int(len(qpos)),
	)
	if errCode < 0 {
		return fmt.Errorf("setState: invalid qpos length \n\thave(%v) "+
			"\n\twant(%v)", len(qpos), m.Nq)
	}

	errCode = C.setQVel(
		m.Data,
		m.Model,
		(*C.double)(unsafe.Pointer(&qvel[0])),
		C.int(len(qvel)),
	)
	if errCode < 0 {
		return fmt.Errorf("setState: invalid qvel length \n\thave(%v) "+
			"\n\twant(%v)", len(qvel), m.Nv)
	}

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
	errCode := C.setControl(
		m.Data,
		m.Model,
		(*C.double)(unsafe.Pointer(&control.RawVector().Data[0])),
		C.int(control.Len()),
	)
	if errCode < 0 {
		return fmt.Errorf("doSimulation: invalid control length "+
			"\n\thave(%v) \n\twant(%v)", control.Len(), m.Nu)
	}

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

// GetBodyCentreOfMass returns the cetnre of mass of the body with
// name bodyName
func (m *MujocoEnv) GetBodyCentreOfMass(bodyName string) (*mat.VecDense,
	error) {
	cBodyName := C.CString(bodyName)
	defer C.free(unsafe.Pointer(cBodyName))
	id := C.mj_name2id(m.Model, C.mjOBJ_BODY, cBodyName)

	if id < 0 {
		return nil, fmt.Errorf("getBodyCentreOfMass: could not find body %v",
			bodyName)
	}

	// here, use a C function to get only the 3 coords then F64SliceC2Go
	var data [3]C.double
	com := F64SliceC2Go(C.getCentreOfMass(id, m.Data, &data[0]), len(data))

	return mat.NewVecDense(len(com), com), nil
}
