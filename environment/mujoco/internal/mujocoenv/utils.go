package mujocoenv

// #cgo CFLAGS: -O2 -I/home/samuel/.mujoco/mujoco200_linux/include -mavx -pthread
// #cgo LDFLAGS: -L/home/samuel/.mujoco/mujoco200_linux/bin -lmujoco200nogl
// #include "mujoco.h"
// #include <stdio.h>
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

func loadXML(file string) (*C.mjModel, *C.mjData, error) {
	// Create MjModel from XML
	modelName := C.CString(file)
	defer C.free(unsafe.Pointer(modelName))
	var err [1000]C.char
	model := C.mj_loadXML(
		modelName,
		nil,
		&err[0],
		C.int(len(err)),
	)
	goErr := C.GoString(&err[0])
	if model == nil || len(goErr) != 0 {
		return nil, nil, fmt.Errorf("could not construct model: %v", goErr)
	}

	// Create the MjData
	data := C.mj_makeData(model)
	if data == nil {
		return nil, nil, fmt.Errorf("could not construct mjData")
	}

	return model, data, nil
}

// F64SliceC2Go converts a copy of a C double array to a Go []float64
//
// See https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
func F64SliceC2Go(array *C.double, len int) []float64 {
	list := (*[1 << 30]float64)(unsafe.Pointer(array))[:len:len]

	newList := make([]float64, len)
	copy(newList, list)

	return newList
}

func StateVector(data *C.mjData, nq, nv int) *mat.VecDense {
	return mat.NewVecDense(
		nq+nv,
		append(
			F64SliceC2Go(data.qpos, nq),
			F64SliceC2Go(data.qvel, nv)...,
		),
	)
}
