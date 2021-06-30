package checkpointer

import (
	"fmt"
	"time"
)

// FileTimer returns a function which will append to a filename the
// number of nanoseconds since January 1, 1970.
func FileTimer(filename, extension string) func() string {
	return func() string {
		return fmt.Sprintf("%v-%v%v", filename, time.Now().UnixNano(),
			extension)
	}
}
