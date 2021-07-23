// Package progressbar implements functionality of printing a progress
// bar to the terminal window
package progressbar

import (
	"fmt"
	"strings"
	"time"
)

// ManualProgressBar implement progress bar functionality that must
// be manually managed. That is, the Display() function must be called
// whenever an updated progress bar should be printed to the screen.
//
// ManualProgressBar does not use concurrency.
type ManualProgressBar struct {
	width           float64
	maxProgress     float64
	currentProgress float64
	bar             strings.Builder
	startTime       time.Time
}

// NewManualProgressBar returns a new ManualProgressBar
func NewManualProgressBar(width, max int) *ManualProgressBar {
	return &ManualProgressBar{
		width:           float64(width),
		maxProgress:     float64(max),
		currentProgress: 0,
		startTime:       time.Now(),
	}
}

// Increment increments the interal progress counter. Each time an
// iteration is performed, Increment should be called.
func (p *ManualProgressBar) Increment() {
	if p.currentProgress < p.maxProgress {
		p.currentProgress++
	}
}

// Display displays the progress bar on the screen. It should only be
// called once.
func (p *ManualProgressBar) Display() {
	p.bar.Reset()
	p.bar.Write([]byte("|"))

	currentProg := p.currentProgress / p.maxProgress * p.width
	for i := 0.0; i < currentProg; i++ {
		p.bar.Write([]byte("█"))
	}
	for i := currentProg; i < p.width; i++ {
		p.bar.Write([]byte(" "))
	}
	p.bar.Write([]byte(fmt.Sprintf("| [%.2f%v | elapsed: %v]",
		p.currentProgress/p.maxProgress*100, "%", time.Since(p.startTime).Truncate(time.Second))))

	fmt.Printf("\n\033[1A\033[K%v", p.bar.String())
}
