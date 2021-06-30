package checkpointer

import "fmt"

// fileEnumerator enumerates filenames
type fileEnumerator struct {
	i         int
	name      string
	extension string
}

// filename returns the name of the next consecutive enumerated file
func (f *fileEnumerator) filename() string {
	f.i++
	return fmt.Sprintf("%v%v%v", f.name, f.i, f.extension)
}

// FilenameEnumerator returns a function which will return filenames
// with a counter integer suffix. Each time the returned function is
//  called, the filename counter suffix will be one higher than on the
// previous call. The filename parameter is the full filename with its
// path, while the extension parameter determines the file extension.
func FilenameEnumerator(start int, filename, extension string) func() string {
	enum := fileEnumerator{i: start, name: filename, extension: extension}

	return enum.filename
}
