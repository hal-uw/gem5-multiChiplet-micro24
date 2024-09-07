/*
 * Copyright (c) 2020 ARM Limited
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

#ifndef __ARCH_ARM_MMU_HH__
#define __ARCH_ARM_MMU_HH__

#include "arch/arm/tlb.hh"
#include "arch/generic/mmu.hh"

#include "params/ArmMMU.hh"

namespace ArmISA {

class MMU : public BaseMMU
{
  protected:
    ArmISA::TLB *
    getDTBPtr() const
    {
        return static_cast<ArmISA::TLB *>(dtb);
    }

    ArmISA::TLB *
    getITBPtr() const
    {
        return static_cast<ArmISA::TLB *>(itb);
    }

  public:
    enum TLBType
    {
        I_TLBS = 0x01,
        D_TLBS = 0x10,
        ALL_TLBS = 0x11
    };

    MMU(const ArmMMUParams &p)
      : BaseMMU(p)
    {}

    bool translateFunctional(ThreadContext *tc, Addr vaddr, Addr &paddr);

    Fault translateFunctional(const RequestPtr &req, ThreadContext *tc,
        BaseTLB::Mode mode, TLB::ArmTranslationType tran_type);

    void invalidateMiscReg(TLBType type = ALL_TLBS);

    template <typename OP>
    void
    flush(const OP &tlbi_op)
    {
        getITBPtr()->flush(tlbi_op);
        getDTBPtr()->flush(tlbi_op);
    }

    template <typename OP>
    void
    iflush(const OP &tlbi_op)
    {
        getITBPtr()->flush(tlbi_op);
    }

    template <typename OP>
    void
    dflush(const OP &tlbi_op)
    {
        getDTBPtr()->flush(tlbi_op);
    }

    uint64_t
    getAttr() const
    {
        return getDTBPtr()->getAttr();
    }
};

template<typename T>
MMU *
getMMUPtr(T *tc)
{
    auto mmu = static_cast<MMU *>(tc->getMMUPtr());
    assert(mmu);
    return mmu;
}


} // namespace ArmISA

#endif // __ARCH_ARM_MMU_HH__
