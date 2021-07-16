#!/bin/bash

# $1:	dir with both, infection, ischaemia, none dirs
# $2:	if 'purge' deletes existent $1/allbut* dirs

if [ $# -gt 1 ] && [ $2 == 'purge' ]
then
	rm -rf $1/allbut*
fi

mkdir $1/allbutboth
mkdir $1/allbutinfection
mkdir $1/allbutischaemia
mkdir $1/allbutnone

cp $1/infection/*.jpg $1/allbutboth/
cp $1/ischaemia/*.jpg $1/allbutboth/
cp $1/none/*.jpg $1/allbutboth/

cp $1/both/*.jpg $1/allbutinfection/
cp $1/ischaemia/*.jpg $1/allbutinfection/
cp $1/none/*.jpg $1/allbutinfection/

cp $1/both/*.jpg $1/allbutischaemia/
cp $1/infection/*.jpg $1/allbutischaemia/
cp $1/none/*.jpg $1/allbutischaemia/

cp $1/both/*.jpg $1/allbutnone/
cp $1/infection/*.jpg $1/allbutnone/
cp $1/ischaemia/*.jpg $1/allbutnone/
