// Package progressbar implements functionality of printing a progress
// bar to the terminal window
package progressbar

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// ProgressBar implements a concurrent progress bar. All updates are
// done in separate GoRoutines so that the progress bar runs
// concurrently with all other processes.
type ProgressBar struct {
	// Width determines the number of characters wide that the progress
	// bar should be
	width float64

	// maxProgress determines the number of times Increment() should
	// be called before the progress bar reaches 100%.
	maxProgress float64

	// currentProgress measures the current progess, equivalently it
	// measures the number of times Increment() was called
	currentProgress float64

	// currentProgressIncrementer is an event channel. When an event
	// appears on this channel, currentProgress is incremented.
	currentProgressIncrementer chan struct{}

	// currentProgressToDisplay is a channel which notifies display()
	// how much current progress to display. The progress bar is then
	// set to <-currentProgressToDisplay/maxProgress percentage of its
	// full width.
	wait                     sync.WaitGroup
	currentProgressToDisplay chan float64

	closeEvent chan struct{}
	closed     bool

	updateEvery       time.Duration
	updateAtIncrement bool

	message chan string
}

// NewProgressBar returns a new progress bar that is width characters
// wide and reaches 100% capacity after max Increment() calls.
func NewProgressBar(width, max int, updateEvery time.Duration,
	updateAtIncrement bool) *ProgressBar {
	pbar := &ProgressBar{
		width:                      float64(width),
		maxProgress:                float64(max),
		currentProgress:            0.0,
		currentProgressIncrementer: make(chan struct{}),
		currentProgressToDisplay:   make(chan float64),
		closeEvent:                 make(chan struct{}),
		closed:                     false,
		updateEvery:                updateEvery,
		updateAtIncrement:          updateAtIncrement,
		message:                    make(chan string),
	}

	// Listen for increment events
	go func() {
		for range pbar.currentProgressIncrementer {
			// Send the currentProgress which should be displayed in
			// the progress bar
			pbar.wait.Add(1)
			pbar.currentProgress++
			pbar.currentProgressToDisplay <- pbar.currentProgress
		}
	}()

	return pbar
}

// AddMessage adds a message after the ProgressBar. If closed is
// called, the progress bar immediately closes, even if there are
// still messages in the queue.
func (p *ProgressBar) AddMessage(message string) {
	go func() {
		p.message <- message
	}()
}

// Increment increments the interal progress counter. Each time an
// iteration is performed, Increment should be called.
func (p *ProgressBar) Increment() {
	go func() {
		if p.currentProgress < p.maxProgress && !p.closed {
			// Send an event signaling that the current progress should
			// be incremented
			p.currentProgressIncrementer <- struct{}{}
		}
	}()
}

// CLose closes the progress bar so that it will no longer display to
// the screen. This function also cleans up any resources the progress
// bar is using.
func (pbar *ProgressBar) Close() {
	pbar.wait.Wait()

	if pbar.closed {
		panic("close: close on closed progress bar")
	}
	close(pbar.closeEvent)
	pbar.closed = true
}

// Display displays the progress bar on the screen. It should only be
// called once.
func (pbar *ProgressBar) Display() {
	go func() {
		currentProgress := pbar.currentProgress
		maxProgress := pbar.maxProgress
		width := pbar.width

		updateEvery := pbar.updateEvery
		tick := time.NewTicker(updateEvery)
		updateAtIncrement := pbar.updateAtIncrement

		message := ""

		var elapsedTime time.Duration = 0 * time.Second
		done := false

		var bar strings.Builder

		for {
			// Update either whenever Increment() is called or every
			// second otherwise.
			select {
			// This case ensures that we are updating whenever Increment()
			// is called if required
			case currentProgress = <-pbar.currentProgressToDisplay:
				pbar.wait.Done()
				if !updateAtIncrement {
					continue
				}

			// Otherwise update every second
			case <-tick.C:
				elapsedTime += updateEvery

			case message = <-pbar.message:

			// Close if a close event is sent
			case <-pbar.closeEvent:
				close(pbar.currentProgressIncrementer)
				close(pbar.currentProgressToDisplay)
				tick.Stop()
				message = ""
				done = true

			default:
				continue
			}

			bar.Reset()
			bar.Write([]byte("|"))

			currentProg := currentProgress / maxProgress * width
			for i := 0.0; i < currentProg; i++ {
				bar.Write([]byte("█"))
			}
			for i := currentProg; i < width; i++ {
				bar.Write([]byte(" "))
			}
			bar.Write([]byte(fmt.Sprintf("| [%.2f%v | elapsed: %v] ",
				currentProgress/maxProgress*100, "%",
				elapsedTime)))

			bar.Write([]byte(message))

			fmt.Printf("\n\033[1A\033[K%v", bar.String())

			if done {
				fmt.Println() // Jump to next line when done
				return
			}
		}
	}()
}
