package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	sourceCode := `
#include <iostream>

int main() {
	int a, b;
	std::cin >> a >> b;
	std::cout << a + b << std::endl;
}
`

	dir, err := os.MkdirTemp("./", "judge-*")
	if err != nil {
		panic(err)
	}

	defer os.RemoveAll(dir)

	// Write source
	sourcePath := filepath.Join(dir, "main.cpp")

	err = os.WriteFile(
		sourcePath,
		[]byte(sourceCode),
		0644,
	)
	if err != nil {
		panic(err)
	}

	// Compile
	compileCmd := exec.Command(
		"g++",
		"main.cpp",
		"-std=c++17",
		"-O2",
		"-o",
		"main",
	)

	compileCmd.Dir = dir

	var compileError bytes.Buffer
	compileCmd.Stderr = &compileError

	if err := compileCmd.Run(); err != nil {
		fmt.Println("CE:")
		fmt.Println(compileError.String())
		return
	}

	// Run
	runCmd := exec.Command("./main")
	runCmd.Dir = dir

	runCmd.Stdin = strings.NewReader("3 4\n")

	var stdout bytes.Buffer
	var stderr bytes.Buffer

	runCmd.Stdout = &stdout
	runCmd.Stderr = &stderr

	err = runCmd.Run()
	if err != nil {
		fmt.Println("RE:")
		fmt.Println(stderr.String())
		return
	}

	fmt.Println("Output:")
	fmt.Println(stdout.String())
}
