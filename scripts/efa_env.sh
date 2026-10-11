# source this only on an EFA DLAMI after confirming actual paths below.
export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/ofi-nccl/lib:/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH:-}
export FI_PROVIDER=efa
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Do not force NCCL_ALGO, NCCL_PROTO, FI_EFA_USE_DEVICE_RDMA or topo XML.
# The OFI plugin selects platform settings; check NET/OFI and provider efa in logs.
