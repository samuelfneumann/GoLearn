// Package progressbar implements functionality of printing a progress
// bar to the terminal window
package progressbar

import (
	"fmt"
	"strings"
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

	// incrementEvent is an event channel. When an even appears on this
	// channel, currentProgress is incremented.
	incrementEvent chan float64

	closeEvent chan struct{}
	closed     bool

	updateEvery       time.Duration
	updateAtIncrement bool
}

// NewProgressBar returns a new progress bar that is width characters
// wide and reaches 100% capacity after max Increment() calls.
func NewProgressBar(width, max int, updateEvery time.Duration,
	updateAtIncrement bool) *ProgressBar {
	return &ProgressBar{
		width:             float64(width),
		maxProgress:       float64(max),
		currentProgress:   0,
		incrementEvent:    make(chan float64),
		closeEvent:        make(chan struct{}),
		closed:            false,
		updateEvery:       updateEvery,
		updateAtIncrement: updateAtIncrement,
	}
}

// Increment increments the interal progress counter. Each time an
// iteration is performed, Increment should be called.
func (p *ProgressBar) Increment() {
	go func() {
		if p.currentProgress < p.maxProgress && !p.closed {
			p.incrementEvent <- p.currentProgress
			p.currentProgress++
		}
	}()
}

// CLose closes the progress bar so that it will no longer display to
// the screen. This function also cleans up any resources the progress
// bar is using.
func (pbar *ProgressBar) Close() {
	if pbar.closed {
		panic("close: close on closed progress bar")
	}
	close(pbar.closeEvent)
	pbar.closed = true
	fmt.Println() // Jump to next line after printed pbar
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

		var elapsedTime time.Duration = 0 * time.Second

		var bar strings.Builder

		for {
			// Update either whenever Increment() is called or every
			// second otherwise.
			select {
			// This case ensures that we are updating whenever Increment()
			// is called if required
			case currentProgress = <-pbar.incrementEvent:
				if !updateAtIncrement {
					continue
				}

			// Otherwise update every second
			case <-tick.C:
				elapsedTime += updateEvery

			// Close if a close event is sent
			case <-pbar.closeEvent:
				close(pbar.incrementEvent)
				tick.Stop()
				return

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
			bar.Write([]byte(fmt.Sprintf("| [%.2f%v | elapsed: %v]",
				currentProgress/maxProgress*100, "%",
				elapsedTime)))

			fmt.Printf("\n\033[1A\033[K%v", bar.String())
		}
	}()
}
