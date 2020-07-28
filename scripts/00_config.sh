if [[ ! -f /.dockerenv ]]; then
	source ${HOME}/.miniconda3/etc/profile.d/conda.sh
	conda activate ${CONDA_ENV:-chainer6}
fi

if [[ $GDB == "1" ]]; then
	PYTHON="gdb -ex run --args ${PYTHON:-python}"

elif [[ $PROFILE == "1" ]]; then
	PYTHON="${PYTHON:-python} -m cProfile -o profile"

else
	PYTHON="${PYTHON:-python}"

fi

SCRIPT="../main.py"
DATA=${DATA:-/home/korsch/Data/info.yml}
DATASET=${DATASET:-CUB200}
MODEL_TYPE=${MODEL_TYPE:-inception}

PARTS=${PARTS:-"GLOBAL"}
SVM_OUTPUT=${SVM_OUTPUT:-"../../.out"}

OPTS="${OPTS} --output ${SVM_OUTPUT}"
OPTS="${OPTS} --feature_model ${MODEL_TYPE}"
