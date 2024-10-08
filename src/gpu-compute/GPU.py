# Copyright (c) 2015-2018 Advanced Micro Devices, Inc.
# All rights reserved.
#
# For use for simulation and test purposes only
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from m5.defines import buildEnv
from m5.params import *
from m5.proxy import *
from m5.SimObject import SimObject

from m5.objects.Bridge import Bridge
from m5.objects.ClockedObject import ClockedObject
from m5.objects.Device import DmaDevice
from m5.objects.HSADevice import HSADevice
from m5.objects.HSADriver import HSADriver
from m5.objects.LdsState import LdsState
from m5.objects.Process import EmulatedDriver

class PrefetchType(Enum): vals = [
    'PF_CU',
    'PF_PHASE',
    'PF_WF',
    'PF_STRIDE',
    'PF_END',
    ]
class GfxVersion(ScopedEnum): vals = [
    'gfx801',
    'gfx803',
    'gfx900',
    ]

class GSPolicyType(Enum): vals = [
    'GSP_RR',
    'GSP_NORM',
    'GSP_LAX',
    'GSP_RRCS'
    ]

class PoolManager(SimObject):
    type = 'PoolManager'
    abstract = True
    cxx_header = "gpu-compute/pool_manager.hh"

    min_alloc = Param.Int(4, 'min number of VGPRs allocated per WF')
    pool_size = Param.Int(2048, 'number of vector registers per SIMD')

# The simple pool manage only allows one workgroup to
# be executing on a CU at any given time.
class SimplePoolManager(PoolManager):
    type = 'SimplePoolManager'
    cxx_class = 'SimplePoolManager'
    cxx_header = "gpu-compute/simple_pool_manager.hh"

## This is for allowing multiple workgroups on one CU
class DynPoolManager(PoolManager):
    type = 'DynPoolManager'
    cxx_class = 'DynPoolManager'
    cxx_header = "gpu-compute/dyn_pool_manager.hh"

class RegisterFile(SimObject):
    type = 'RegisterFile'
    cxx_class = 'RegisterFile'
    cxx_header = 'gpu-compute/register_file.hh'

    simd_id = Param.Int(-1, 'SIMD ID associated with this Register File')
    num_regs = Param.Int(2048, 'number of registers in this RF')
    wf_size = Param.Int(64, 'Wavefront size (in work items)')

class ScalarRegisterFile(RegisterFile):
    type = 'ScalarRegisterFile'
    cxx_class = 'ScalarRegisterFile'
    cxx_header = 'gpu-compute/scalar_register_file.hh'

class VectorRegisterFile(RegisterFile):
    type = 'VectorRegisterFile'
    cxx_class = 'VectorRegisterFile'
    cxx_header = 'gpu-compute/vector_register_file.hh'

class RegisterManager(SimObject):
    type = 'RegisterManager'
    cxx_class = 'RegisterManager'
    cxx_header = 'gpu-compute/register_manager.hh'

    policy = Param.String("static", "Register Manager Policy")
    vrf_pool_managers = VectorParam.PoolManager('VRF Pool Managers')
    srf_pool_managers = VectorParam.PoolManager('SRF Pool Managers')

class Wavefront(SimObject):
    type = 'Wavefront'
    cxx_class = 'Wavefront'
    cxx_header = 'gpu-compute/wavefront.hh'

    simdId = Param.Int('SIMD id (0-ComputeUnit.num_SIMDs)')
    wf_slot_id = Param.Int('wavefront id (0-ComputeUnit.max_wfs)')
    wf_size = Param.Int(64, 'Wavefront size (in work items)')
    max_ib_size = Param.Int(13, 'Maximum size (in number of insts) of the '
                            'instruction buffer (IB).')

# Most of the default values here are obtained from the
# AMD Graphics Core Next (GCN) Architecture whitepaper.
class ComputeUnit(ClockedObject):
    type = 'ComputeUnit'
    cxx_class = 'ComputeUnit'
    cxx_header = 'gpu-compute/compute_unit.hh'

    wavefronts = VectorParam.Wavefront('Number of wavefronts')
    # Wavefront size is 64. This is configurable, however changing
    # this value to anything other than 64 will likely cause errors.
    wf_size = Param.Int(64, 'Wavefront size (in work items)')
    num_barrier_slots = Param.Int(16, 'Number of barrier slots in a CU')
    num_SIMDs = Param.Int(4, 'number of SIMD units per CU')
    num_scalar_cores = Param.Int(1, 'number of Scalar cores per CU')
    num_scalar_mem_pipes = Param.Int(1, 'number of Scalar memory pipelines '\
                                     'per CU')
    simd_width = Param.Int(16, 'width (number of lanes) per SIMD unit')

    operand_network_length = Param.Int(1, 'number of pipe stages of operand '\
                                          'network')

    spbypass_pipe_length = Param.Int(4, 'vector ALU Single Precision bypass '\
                                        'latency')

    dpbypass_pipe_length = Param.Int(4, 'vector ALU Double Precision bypass '\
                                        'latency')
    scalar_pipe_length = Param.Int(1, 'number of pipe stages per scalar ALU')
    issue_period = Param.Int(4, 'number of cycles per issue period')

    vrf_gm_bus_latency = Param.Int(1, 'number of cycles per use of VRF to '\
                                      'GM bus')
    srf_scm_bus_latency = Param.Int(1, 'number of cycles per use of SRF '\
                                       'to Scalar Mem bus')
    vrf_lm_bus_latency = Param.Int(1, 'number of cycles per use of VRF to '\
                                      'LM bus')

    num_global_mem_pipes = Param.Int(1,'number of global memory pipes per CU')
    num_shared_mem_pipes = Param.Int(1,'number of shared memory pipes per CU')
    n_wf = Param.Int(10, 'Number of wavefront slots per SIMD')
    mem_req_latency = Param.Int(50, "Latency for request from the cu to ruby."\
                                " Represents the pipeline to reach the TCP "\
                                "and specified in GPU clock cycles")
    mem_resp_latency = Param.Int(50, "Latency for responses from ruby to the "\
                                 "cu. Represents the pipeline between the "\
                                 "TCP and cu as well as TCP data array "\
                                 "access. Specified in GPU clock cycles")
    system = Param.System(Parent.any, "system object")
    cu_id = Param.Int('CU id')
    vrf_to_coalescer_bus_width = Param.Int(64, "VRF->Coalescer data bus "\
                                           "width in bytes")
    coalescer_to_vrf_bus_width = Param.Int(64, "Coalescer->VRF data bus "\
                                           "width  in bytes")

    memory_port = VectorRequestPort("Port to the memory system")
    translation_port = VectorRequestPort('Port to the TLB hierarchy')
    sqc_port = RequestPort("Port to the SQC (I-cache")
    sqc_tlb_port = RequestPort("Port to the TLB for the SQC (I-cache)")
    scalar_port = RequestPort("Port to the scalar data cache")
    scalar_tlb_port = RequestPort("Port to the TLB for the scalar data cache")
    gmTokenPort = RequestPort("Port to the GPU coalesecer for sharing tokens")

    perLaneTLB = Param.Bool(False, "enable per-lane TLB")
    prefetch_depth = Param.Int(0, "Number of prefetches triggered at a time"\
                               "(0 turns off prefetching)")
    prefetch_stride = Param.Int(1, "Fixed Prefetch Stride (1 means next-page)")
    prefetch_prev_type = Param.PrefetchType('PF_PHASE', "Prefetch the stride "\
                                            "from last mem req in lane of "\
                                            "CU|Phase|Wavefront")
    execPolicy = Param.String("OLDEST-FIRST", "WF execution selection policy");
    debugSegFault = Param.Bool(False, "enable debugging GPU seg faults")
    functionalTLB = Param.Bool(False, "Assume TLB causes no delay")

    localMemBarrier = Param.Bool(False, "Assume Barriers do not wait on "\
                                        "kernel end")

    countPages = Param.Bool(False, "Generate per-CU file of all pages "\
                            "touched and how many times")
    scalar_mem_queue_size = Param.Int(32, "Number of entries in scalar "\
                                      "memory pipeline's queues")
    global_mem_queue_size = Param.Int(256, "Number of entries in the global "
                                      "memory pipeline's queues")
    local_mem_queue_size = Param.Int(256, "Number of entries in the local "
                                      "memory pipeline's queues")
    max_wave_requests = Param.Int(64, "number of pending vector memory "\
                                      "requests per wavefront")
    max_cu_tokens = Param.Int(4, "Maximum number of tokens, i.e., the number"\
                            " of instructions that can be sent to coalescer")
    ldsBus = Bridge() # the bridge between the CU and its LDS
    ldsPort = RequestPort("The port that goes to the LDS")
    localDataStore = Param.LdsState("the LDS for this CU")

    vector_register_file = VectorParam.VectorRegisterFile("Vector register "\
                                                          "file")

    scalar_register_file = VectorParam.ScalarRegisterFile("Scalar register "\
                                                          "file")
    out_of_order_data_delivery = Param.Bool(False, "enable OoO data delivery"
                                            " in the GM pipeline")
    register_manager = Param.RegisterManager("Register Manager")
    fetch_depth = Param.Int(2, 'number of i-cache lines that may be '
                            'buffered in the fetch unit.')

class Shader(ClockedObject):
    type = 'Shader'
    cxx_class = 'Shader'
    cxx_header = 'gpu-compute/shader.hh'
    CUs = VectorParam.ComputeUnit('Number of compute units')
    gpu_cmd_proc = Param.GPUCommandProcessor('Command processor for GPU')
    dispatcher = Param.GPUDispatcher('GPU workgroup dispatcher')
    n_wf = Param.Int(10, 'Number of wavefront slots per SIMD')
    impl_kern_launch_acq = Param.Bool(True, """Insert acq packet into
                                         ruby at kernel launch""")
    impl_kern_end_rel = Param.Bool(False, """Insert rel packet into
                                         ruby at kernel end""")
    globalmem = Param.MemorySize('64kB', 'Memory size')
    timing = Param.Bool(False, 'timing memory accesses')

    cpu_pointer = Param.BaseCPU(NULL, "pointer to base CPU")
    translation = Param.Bool(False, "address translation");
    timer_period = Param.Clock('10us', "system timer period")
    idlecu_timeout = Param.Tick(0, "Idle CU watchdog timeout threshold")
    max_valu_insts = Param.Int(0, "Maximum vALU insts before exiting")
    gpu_num = Param.Int(1, "GPU Number");
    reset_reg_pool = Param.Bool(False, "Reset register pool allocations on "
                                "kernel boundaries.")

class GPUComputeDriver(HSADriver):
    type = 'GPUComputeDriver'
    cxx_header = 'gpu-compute/gpu_compute_driver.hh'
    num_gpus = Param.Int(1, 'Number of GPUs')
    isdGPU = Param.Bool(False, 'Driver is for a dGPU')
    gfxVersion = Param.GfxVersion('gfx801', 'ISA of gpu to model')

class GPUDispatcher(SimObject):
    type = 'GPUDispatcher'
    cxx_header = 'gpu-compute/dispatcher.hh'
    threshold = Param.Float(0.75, "Threshold to request more work")
    gpu_id = Param.Int(1, "GPU ID")

class GPUCommandProcessor(HSADevice):
    type = 'GPUCommandProcessor'
    cxx_header = 'gpu-compute/gpu_command_processor.hh'
    dispatcher = Param.GPUDispatcher('workgroup dispatcher for the GPU')

class StorageClassType(Enum): vals = [
    'SC_SPILL',
    'SC_GLOBAL',
    'SC_GROUP',
    'SC_PRIVATE',
    'SC_READONLY',
    'SC_KERNARG',
    'SC_ARG',
    'SC_NONE',
    ]

class GlobalScheduler(DmaDevice):
    type = 'GlobalScheduler'
    cxx_class = 'GlobalScheduler'
    cxx_header = 'gpu-compute/global_scheduler.hh'
    pioAddr = Param.Addr("address")
    pioDelay = Param.Tick(10, 'pio delay')
    hsapp = VectorParam.HSAPacketProcessor([], """HSAPacketProcessor controlled
                                                by this scheduler""")
    #shader_list = VectorParam.Shader([],'Shader controlled by this scheduler')
    device = VectorParam.GPUCommandProcessor([],"""Command Processor controlled
                                                by this scheduler""")
    dispatcher = VectorParam.GPUDispatcher([],"""Dispatcher controlled by this
                                              scheduler""")
    num_gpus = Param.Int(1,'number of gpus')
    n_cu = Param.Int(4,'number of compute units')
    n_wf = Param.Int(10, 'number of wf per simd')
    num_SIMDs = Param.Int(4, 'number of simd per cu')
    wf_size = Param.Int(64, 'Wavefront size (in workitems)')
    sched_policy = Param.GSPolicyType("GSP_NORM", 'Global Scheduling policy')
    outdir = Param.String("m5out", "Output directory of this run")
    num_sched_gpu = Param.Int(2, 'number of chiplets kernel is scheduled across')
    default_acq_rel = Param.Bool(False, "default acq/rel behabiour")
    num_tccs = Param.Int(8,'number of TCCs')
