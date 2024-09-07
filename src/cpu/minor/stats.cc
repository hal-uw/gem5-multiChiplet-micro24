/*
 * Copyright (c) 2012-2014 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu/minor/stats.hh"

namespace Minor
{

MinorStats::MinorStats(BaseCPU *base_cpu)
    : Stats::Group(base_cpu),
    ADD_STAT(numInsts, "Number of instructions committed"),
    ADD_STAT(numOps, "Number of ops (including micro ops) committed"),
    ADD_STAT(numDiscardedOps,
             "Number of ops (including micro ops) which were discarded before "
             "commit"),
    ADD_STAT(numFetchSuspends,
             "Number of times Execute suspended instruction fetching"),
    ADD_STAT(quiesceCycles,
             "Total number of cycles that CPU has spent quiesced or waiting "
             "for an interrupt"),
    ADD_STAT(cpi, "CPI: cycles per instruction"),
    ADD_STAT(ipc, "IPC: instructions per cycle"),
    ADD_STAT(committedInstType, "Class of committed instruction")
{
    quiesceCycles.prereq(quiesceCycles);

    cpi.precision(6);
    cpi = base_cpu->baseStats.numCycles / numInsts;

    ipc.precision(6);
    ipc = numInsts / base_cpu->baseStats.numCycles;

    committedInstType
        .init(base_cpu->numThreads, Enums::Num_OpClass)
        .flags(Stats::total | Stats::pdf | Stats::dist);
    committedInstType.ysubnames(Enums::OpClassStrings);
}

};
