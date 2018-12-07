#!/usr/bin/env bash

echo "Runing before function"

function fun_A() {
    echo "fucntion A"
}

function fun_B() {
    echo "fuction B"
}

echo "After all functions"


for var in "$@"
do
    echo "$var"
    $var
done

