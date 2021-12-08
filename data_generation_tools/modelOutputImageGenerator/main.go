package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
)

func main() {
	size := 128
	orderFix := getOrderFix()
	lines, err := os.ReadFile("images1.txt")
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	img := image.NewNRGBA(image.Rect(0, 0, size, size))
	linesStr := string(lines)
	nums := strings.Split(linesStr, "\n")
	for index, str := range nums {
		if index == size*size {
			break
		}
		isTrees, _ := strconv.Atoi(str)
		print("num = " + strconv.Itoa(index) + "\n")
		index = orderFix[index]
		if isTrees == 1 {
			img.Set(index%size, index/size, color.NRGBA{0, 255, 0, 255})
		} else {
			//img.Set(index%size, index/size, color.NRGBA{100, 100, 100, 255})
		}
	}
	outFile, err := os.Create("image1.png")
	if err != nil {
		log.Fatal(err)
	}
	defer outFile.Close()
	png.Encode(outFile, img)
}

func getOrderFix() []int {
	// For
	const s = 128
	var a []string
	var b []string
	for i := 0; i < s*s; i++ {
		a = append(a, strconv.Itoa(i))
		b = append(b, strconv.Itoa(i))
	}
	sort.Strings(a)
	f, err := os.OpenFile("reordering.txt", os.O_WRONLY, 0777)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	var out []int
	index := 0
	for i := 0; i < s*s; i++ {
		for j := index; j < s*s+index; j++ {
			if b[j%(s*s)] == a[i] {
				index = j % (s * s)
				break
			}
			if j == s*s+index-1 {
				index = -1
			}
		}
		out = append(out, index)
		_, err := f.WriteString(strconv.Itoa(i) + " -> " + strconv.Itoa(index) + "\n")
		if err != nil {
			log.Fatal(err)
		}
		println(strconv.Itoa(i) + " -> " + strconv.Itoa(index))
	}
	return out
}
