package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"

	// Blank imports needed for registering agents with agent package
	// to enable TypedConfigList's
	"github.com/samuelfneumann/gogym"
	_ "github.com/samuelfneumann/golearn/agent/linear/continuous/actorcritic"
	_ "github.com/samuelfneumann/golearn/agent/linear/discrete/esarsa"
	_ "github.com/samuelfneumann/golearn/agent/linear/discrete/qlearning"
	_ "github.com/samuelfneumann/golearn/agent/nonlinear/continuous/vanillaac"
	_ "github.com/samuelfneumann/golearn/agent/nonlinear/continuous/vanillapg"
	_ "github.com/samuelfneumann/golearn/agent/nonlinear/discrete/deepq"

	"github.com/samuelfneumann/golearn/experiment"
	"github.com/samuelfneumann/golearn/experiment/checkpointer"
	"github.com/samuelfneumann/golearn/experiment/tracker"
)

// func main() {
// 	init, err := initwfn.NewGlorotN(math.Sqrt(2.0))
// 	if err != nil {
// 		panic(err)
// 	}

// 	pSolver, err := solver.NewDefaultAdam(0.1, 1)
// 	if err != nil {
// 		panic(err)
// 	}

// 	vSolver, err := solver.NewDefaultAdam(0.1, 1)
// 	if err != nil {
// 		panic(err)
// 	}

// 	config := vanillaac.NewGaussianTreeMLPConfigList(
// 		[][]int{{5, 5}},
// 		[][]bool{{true, true}},
// 		[][]*network.Activation{{network.ReLU(), network.ReLU()}},
// 		[][][]int{{{5, 5}, {5, 5}}},
// 		[][][]bool{{{true, true}, {true, true}}},
// 		[][][]*network.Activation{
// 			{
// 				{network.ReLU(), network.ReLU()},
// 				{network.ReLU(), network.ReLU()},
// 			},
// 		},
// 		[][]int{{5, 5}},
// 		[][]bool{{true, true}},
// 		[][]*network.Activation{{network.ReLU(), network.ReLU()}},

// 		[]*initwfn.InitWFn{init},

// 		// Batch size 1 -> we don't train in mini-batches of data in RL
// 		[]*solver.Solver{pSolver},
// 		[]*solver.Solver{vSolver},

// 		[]int{1},
// 		[]expreplay.Config{
// 			{
// 				RemoveMethod:      expreplay.Fifo,
// 				SampleMethod:      expreplay.Uniform,
// 				RemoveSize:        1,
// 				SampleSize:        32,
// 				MaxReplayCapacity: 1_000_000,
// 				MinReplayCapacity: 32,
// 			},
// 		},
// 	)

// 	f, err := os.Create("VAC-Gaussian.json")
// 	if err != nil {
// 		panic(err)
// 	}
// 	defer f.Close()
// 	enc := json.NewEncoder(f)
// 	enc.SetIndent("", "\t")
// 	enc.Encode(config)
// }

func main() {
	expFile, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	dec := json.NewDecoder(expFile)

	var expConf experiment.Config
	err = dec.Decode(&expConf)
	if err != nil {
		panic(fmt.Sprintf("could not decode experiment config: %v",
			err))
	}
	expFile.Close()

	numSettings := int64(expConf.AgentConfig.Len())
	hpIndex, err := strconv.ParseInt(os.Args[2], 0, 0)
	if err != nil {
		panic(err)
	}
	run := uint64(hpIndex / numSettings)

	// Print some information about the experiment
	fmt.Println("=== Experiment Starting")
	fmt.Printf("\t Experiment Type:\t\t%v \n", expConf.Type)
	fmt.Printf("\t Experiment Total Steps:\t%v \n", expConf.MaxSteps)
	fmt.Printf("\t Total Configurations: \t\t%v\n", numSettings)
	fmt.Printf("\t Configuration Index: \t\t%v\n", hpIndex%numSettings)
	fmt.Printf("\t Run: \t\t\t\t%v\n", run)
	fmt.Println()
	fmt.Printf("\t Environment: \t\t\t%v\n", expConf.EnvConfig.Environment)
	fmt.Printf("\t Environment Configuration: \t%v\n", expConf.EnvConfig)
	fmt.Println()
	fmt.Printf("\t Agent: \t\t\t%v\n", expConf.AgentConfig.Type)
	fmt.Printf("\t Agent Configuration: \t\t%v\n",
		expConf.AgentConfig.At(int(hpIndex)))
	fmt.Println()

	// Filenames of data to save
	returnFilename := fmt.Sprintf(
		"return_%v_%v_run%v.bin",
		expConf.AgentConfig.Type,
		expConf.EnvConfig.Environment,
		run,
	)
	epLengthFilename := fmt.Sprintf(
		"epLength_%v_%v_run%v.bin",
		expConf.AgentConfig.Type,
		expConf.EnvConfig.Environment,
		run,
	)

	// Create trackers to track and save data from experiment
	trackers := []tracker.Tracker{
		tracker.NewReturn(returnFilename),
		tracker.NewEpisodeLength(epLengthFilename),
	}

	// Don't checkpoint agents
	var checkpointers []checkpointer.Checkpointer = nil

	exp, err := expConf.CreateExp(int(hpIndex), run, trackers, checkpointers)
	if err != nil {
		log.Printf("Error creating experiment: %v\n", err)
		log.Println("Terminating...")
	}
	if err := exp.Run(); err != nil {
		log.Printf("Error in running experiment: %v\n", err)
		log.Println("Terminating...")
	}
	exp.Save()

	// LoadData -> should be int or float specified...
	data := tracker.LoadFData(returnFilename)
	fmt.Println(data)

	// Clean up GoGym package before leaving
	gogym.Close()
}
