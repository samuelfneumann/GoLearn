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
	// measures the number of times Increment() was called.
	// currentProgressIncrementer is an event channel. When an event
	// appears on this channel, the current progress should be
	// incremented
	currentProgress            float64
	currentProgressIncrementer chan struct{}

	// progressToDisplay is a channel along which the progress of the
	// progress bar is sent to be displayed.
	wait              sync.WaitGroup
	progressToDisplay chan float64

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
		currentProgress:            0,
		currentProgressIncrementer: make(chan struct{}),
		progressToDisplay:          make(chan float64),
		closeEvent:                 make(chan struct{}),
		closed:                     false,
		updateEvery:                updateEvery,
		updateAtIncrement:          updateAtIncrement,
		message:                    make(chan string),
	}

	// Listen for increment events
	go func() {
		for range pbar.currentProgressIncrementer {
			pbar.currentProgress++
		}
	}()

	return pbar
}

// Increment increments the interal progress counter. Each time an
// iteration is performed, Increment should be called.
func (p *ProgressBar) Increment() {
	p.wait.Add(1)
	go func() {
		if p.currentProgress < p.maxProgress && !p.closed {
			// Hang at incrementing, ensuring the progress bar has been
			// incremented, before updating the display.
			p.currentProgressIncrementer <- struct{}{}
			p.progressToDisplay <- p.currentProgress
		}
		p.wait.Done()
	}()
}

// AddMessage adds a message after the ProgressBar. If closed is
// called, the progress bar immediately closes, even if there are
// still messages in the queue.
func (p *ProgressBar) AddMessage(message string) {
	go func() {
		p.message <- message
	}()
}

// Close closes the progress bar so that it will no longer display to
// the screen. This function also cleans up any resources the progress
// bar is using.
func (pbar *ProgressBar) Close() {
	pbar.wait.Wait()

	// Do a final refresh of the display
	pbar.progressToDisplay <- pbar.currentProgress
	pbar.message <- ""

	// Close the progress bar
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

		var elapsedTime time.Duration = 0 * time.Second
		var message string

		var bar strings.Builder

		for {
			// Update either whenever Increment() is called or every
			// second otherwise.
			select {
			// This case ensures that we are updating whenever Increment()
			// is called if required
			case currentProgress = <-pbar.progressToDisplay:
				if !updateAtIncrement {
					continue
				}

			// Otherwise update every second
			case <-tick.C:
				elapsedTime += updateEvery

			case message = <-pbar.message:

			// Close if a close event is sent
			case <-pbar.closeEvent:
				close(pbar.progressToDisplay)
				close(pbar.currentProgressIncrementer)
				tick.Stop()
				fmt.Println() // Jump to next line
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
			bar.Write([]byte(fmt.Sprintf("| [%.2f%v | elapsed: %v] ",
				currentProgress/maxProgress*100, "%",
				elapsedTime)))

			bar.Write([]byte(message))

			fmt.Printf("\n\033[1A\033[K%v", bar.String())
		}
	}()
}
