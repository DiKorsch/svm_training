#!/usr/bin/env bash

source 00_config.sh

$PYTHON $SCRIPT \
	${DATA} \
	${DATASET} \
	${PARTS} \
	${OPTS} \
	$@
