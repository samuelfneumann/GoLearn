package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	// Blank imports needed for registering agents with agent package
	// to enable TypedConfigList's
	_ "github.com/samuelfneumann/golearn/agent/linear/continuous/actorcritic"
	_ "github.com/samuelfneumann/golearn/agent/linear/discrete/esarsa"
	_ "github.com/samuelfneumann/golearn/agent/linear/discrete/qlearning"
	_ "github.com/samuelfneumann/golearn/agent/nonlinear/continuous/vanillapg"
	_ "github.com/samuelfneumann/golearn/agent/nonlinear/discrete/deepq"

	"github.com/samuelfneumann/golearn/experiment"
	"github.com/samuelfneumann/golearn/experiment/checkpointer"
	"github.com/samuelfneumann/golearn/experiment/tracker"
)

func main() {
	expFile, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	dec := json.NewDecoder(expFile)

	var expConf experiment.Config
	err = dec.Decode(&expConf)
	if err != nil {
		panic(err)
	}
	expFile.Close()

	numSettings := int64(expConf.AgentConf.Len())
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
	fmt.Printf("\t Run: \t\t\t\t%v\n", run)
	fmt.Printf("\t Configuration Index: \t\t%v\n", hpIndex%numSettings)
	fmt.Println()
	fmt.Printf("\t Environment: \t\t\t%v\n", expConf.EnvConf.Environment)
	fmt.Printf("\t Environment Configuration: \t%v\n", expConf.EnvConf)
	fmt.Println()
	fmt.Printf("\t Agent: \t\t\t%v\n", expConf.AgentConf.Type)
	fmt.Printf("\t Agent Configuration: \t\t%v\n",
		expConf.AgentConf.At(int(hpIndex)))
	fmt.Println()

	// Filenames of data to save
	returnFilename := fmt.Sprintf(
		"return_%v_%v_run%v.bin",
		expConf.AgentConf.Type,
		expConf.EnvConf.Environment,
		run,
	)
	epLengthFilename := fmt.Sprintf(
		"epLength_%v_%v_run%v.bin",
		expConf.AgentConf.Type,
		expConf.EnvConf.Environment,
		run,
	)

	// Create trackers to track and save data from experiment
	trackers := []tracker.Tracker{
		tracker.NewReturn(returnFilename),
		tracker.NewEpisodeLength(epLengthFilename),
	}

	// Don't checkpoint agents
	var checkpointers []checkpointer.Checkpointer = nil

	exp := expConf.CreateExp(int(hpIndex), run, trackers, checkpointers)
	exp.Run()
	exp.Save()

	// LoadData -> should be int or float specified...
	data := tracker.LoadFData(returnFilename)
	fmt.Println(data)
}
