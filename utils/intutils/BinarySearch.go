// Package tree implements tree data structures for int
package intutils

import (
	"fmt"
	"strings"
	"sync"
)

// BinarySearchNode is a single node that composes the tree
type BinarySearchNode struct {
	value int
	left  *BinarySearchNode //left
	right *BinarySearchNode //right
}

// Value returns the value of the node
func (b *BinarySearchNode) Value() int {
	return b.value
}

// BinarySearch the binary search tree of ints
type BinarySearch struct {
	root *BinarySearchNode
	lock sync.RWMutex
	len  int
}

// NewBinarySearch returns a new BinarySearch tree
func NewBinarySearch() *BinarySearch {
	return &BinarySearch{}
}

// Len returns the number of elements in the tree
func (b *BinarySearch) Len() int {
	return b.len
}

// Root returns the root of the tree
func (b *BinarySearch) Root() *BinarySearchNode {
	return b.root
}

// Insert inserts value into the tree
func (bst *BinarySearch) Insert(value int) {
	bst.lock.Lock()
	defer bst.lock.Unlock()
	bst.len++

	n := &BinarySearchNode{value, nil, nil}
	if bst.root == nil {
		bst.root = n
	} else {
		insertBinarySearchNode(bst.root, n)
	}
}

// insertBinarySearchNode inserts newBinarySearchNode into the tree
// under the node node
func insertBinarySearchNode(node, newBinarySearchNode *BinarySearchNode) {
	if newBinarySearchNode.value < node.value {
		if node.left == nil {
			node.left = newBinarySearchNode
		} else {
			insertBinarySearchNode(node.left, newBinarySearchNode)
		}
	} else {
		if node.right == nil {
			node.right = newBinarySearchNode
		} else {
			insertBinarySearchNode(node.right, newBinarySearchNode)
		}
	}
}

// InOrderTraverse visits all nodes with in-order traversing
func (bst *BinarySearch) InOrderTraverse(f func(int)) {
	bst.lock.RLock()
	defer bst.lock.RUnlock()

	inOrderTraverse(bst.root, f)
}

// inOrderTraverse vistsis all nodes with in-order traversing, applying
// function f to each node's value
func inOrderTraverse(n *BinarySearchNode, f func(int)) {
	if n != nil {
		inOrderTraverse(n.left, f)
		f(n.value)
		inOrderTraverse(n.right, f)
	}
}

// PreOrderTraverse visits all nodes with pre-order traversing
func (bst *BinarySearch) PreOrderTraverse(f func(int)) {
	bst.lock.Lock()
	defer bst.lock.Unlock()

	preOrderTraverse(bst.root, f)
}

// preOrderTraverse vistsis all nodes with pre-order traversing,
// applying function f to each node's value
func preOrderTraverse(n *BinarySearchNode, f func(int)) {
	if n != nil {
		f(n.value)
		preOrderTraverse(n.left, f)
		preOrderTraverse(n.right, f)
	}
}

// PostOrderTraverse visits all nodes with post-order traversing
func (bst *BinarySearch) PostOrderTraverse(f func(int)) {
	bst.lock.Lock()
	defer bst.lock.Unlock()

	postOrderTraverse(bst.root, f)
}

// postOrderTraverse vistsis all nodes with post-order traversing,
// applying function f to each node's value
func postOrderTraverse(n *BinarySearchNode, f func(int)) {
	if n != nil {
		postOrderTraverse(n.left, f)
		postOrderTraverse(n.right, f)
		f(n.value)
	}
}

// Min returns the int with min value stored in the tree
func (bst *BinarySearch) Min() (int, error) {
	bst.lock.RLock()
	defer bst.lock.RUnlock()

	n := bst.root
	if n == nil {
		return 0, fmt.Errorf("min: no nodes exist in tree")
	}
	for {
		if n.left == nil {
			return n.value, nil
		}
		n = n.left
	}
}

// Max returns the int with max value stored in the tree
func (bst *BinarySearch) Max() (int, error) {
	bst.lock.RLock()
	defer bst.lock.RUnlock()

	n := bst.root
	if n == nil {
		return 0, fmt.Errorf("max: no nodes exist in tree")
	}
	for {
		if n.right == nil {
			return n.value, nil
		}
		n = n.right
	}
}

// Search returns true if value exists in the tree
func (bst *BinarySearch) Search(value int) bool {
	bst.lock.RLock()
	defer bst.lock.RUnlock()

	return search(bst.root, value)
}

// search returns true if value exists in the tree under node n
func search(n *BinarySearchNode, value int) bool {
	if n == nil {
		return false
	}
	if value < n.value {
		return search(n.left, value)
	}
	if value > n.value {
		return search(n.right, value)
	}
	return true
}

// Remove removes value from the tree
func (bst *BinarySearch) Remove(value int) {
	bst.lock.Lock()
	defer bst.lock.Unlock()

	node := remove(bst.root, value)
	if node != nil {
		bst.len--
	}
}

// Remove removes value from the tree under node
func remove(node *BinarySearchNode, value int) *BinarySearchNode {
	if node == nil {
		return nil
	}
	if value < node.value {
		node.left = remove(node.left, value)
		return node
	}
	if value > node.value {
		node.right = remove(node.right, value)
		return node
	}

	if node.left == nil && node.right == nil {
		node = nil
		return nil
	}
	if node.left == nil {
		node = node.right
		return node
	}
	if node.right == nil {
		node = node.left
		return node
	}

	// Find the node that has the smallest value greater than the value
	// we are removing and place it at the position of the node we
	// are removing
	leftmostrightside := node.right
	for {
		if leftmostrightside != nil && leftmostrightside.left != nil {
			leftmostrightside = leftmostrightside.left
		} else {
			break
		}
	}
	node.value = leftmostrightside.value
	node.right = remove(node.right, node.value)
	return node
}

// String implements the fmt.Stringer interface
func (bst *BinarySearch) String() string {
	bst.lock.Lock()
	defer bst.lock.Unlock()

	var builder strings.Builder

	builder.WriteString("-------------------------------------------\n")
	stringify(bst.root, 0, &builder)
	builder.WriteString("-------------------------------------------\n")

	return builder.String()
}

// stringify converts a BinarySearchNode at a specific level into a
// string
func stringify(n *BinarySearchNode, level int, builder *strings.Builder) {
	if n != nil {
		format := ""
		for i := 0; i < level; i++ {
			format += "       "
		}
		format += "---[ "
		level++
		stringify(n.left, level, builder)
		builder.WriteString(fmt.Sprintf(format+"%d\n", n.value))
		stringify(n.right, level, builder)
	}
}
