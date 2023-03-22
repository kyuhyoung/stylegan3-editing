docker_name=u_18_cpp_7_5_py_3_6_cuda_11_1_cudnn_8_0_cv_4_7_torch_1_10_tv_0_11_tb_2_5
dir_cur=/workspace/${PWD##*/}
dir_data=/mnt/d/data
#: << 'END'
#################################################################################################
#   docker build
cd docker_file/; docker build --force-rm --shm-size=64g -t ${docker_name} -f Dockerfile_${docker_name} .; cd -
#END

#: << 'END'
#################################################################################################
#   docker info.
#docker run --rm -it -w $PWD -v $PWD:$PWD ${docker_name} bash docker_file/extract_docker_info.sh
docker run --rm -it -w $PWD -v $PWD:$PWD ${docker_name} sh -c ". ~/.bashrc && . docker_file/extract_docker_info.sh"
#END

#: << 'END'
#################################################################################################
#   docker run
docker run --rm -it --shm-size=64g --gpus '"device=0"' -e DISPLAY=$DISPLAY -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} bash
#END
