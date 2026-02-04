#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"
#include "operators/transpose.h"

namespace infini
{
    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }


    void GraphObj::optimize() {
        //      =================================== 作业 ===================================
        //     TODO: 设计一个算法来实现指定的图优化规则
        //     图优化规则如下：
        //     1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        //     2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        //     =================================== 作业 ===================================
        // 检查 Transpose 是否仅仅交换最后两维 
        auto isSwapLastTwo = [](const vector<int> &perm) {
            int r = perm.size();
            if (r < 2) return false;
            for (int i = 0; i < r - 2; ++i) {
                if (perm[i] != i) return false;
            }
            return (perm[r - 1] == r - 2 && perm[r - 2] == r - 1);
        };

        bool changed = true;
        while (changed) {
            changed = false;

            // 核心优化逻辑遍历
            for (auto &op : ops) {
                // Transpose 融入 MatMul
                if (op->getOpType() == OpType::MatMul) {
                    auto matmul = std::dynamic_pointer_cast<MatmulObj>(op);
                    for (int i = 0; i < 2; ++i) {
                        auto input = matmul->getInputs()[i];
                        auto prevOp = input->getSource();
                        if (prevOp && prevOp->getOpType() == OpType::Transpose) {
                            auto trans = std::dynamic_pointer_cast<TransposeObj>(prevOp);
                            if (isSwapLastTwo(trans->getPermute())) {
                                // 逻辑融合：修改 MatMul 属性
                                if (i == 0) matmul->setTransA(!matmul->getTransA());
                                else matmul->setTransB(!matmul->getTransB());
                                
                                // 重新连线
                                auto originIn = trans->getInputs()[0];
                                input->removeTarget(op); // 旧 Tensor 释放 op
                                originIn->addTarget(op); // 新 Tensor 绑定 op
                                op->replaceInput(input, originIn); 
                                changed = true;
                            }
                        }
                    }
                }

                // 连续冗余 Transpose 消除 
                if (op->getOpType() == OpType::Transpose) {
                    auto trans2 = std::dynamic_pointer_cast<TransposeObj>(op);
                    auto inputT2 = trans2->getInputs()[0];
                    auto prevOp = inputT2->getSource();
                    if (prevOp && prevOp->getOpType() == OpType::Transpose) {
                        auto trans1 = std::dynamic_pointer_cast<TransposeObj>(prevOp);
                        if (isSwapLastTwo(trans1->getPermute()) && isSwapLastTwo(trans2->getPermute())) {
                            auto originIn = trans1->getInputs()[0];
                            auto outputT2 = trans2->getOutput();
                            
                            // 让所有下游算子绕过这两个 Transpose，直接看最初的输入
                            auto targets = outputT2->getTargets();
                            for (auto &nextOp : targets) {
                                outputT2->removeTarget(nextOp);
                                originIn->addTarget(nextOp);
                                nextOp->replaceInput(outputT2, originIn);
                            }
                            changed = true;
                        }
                    }
                }
            }

            // 彻底移除被优化掉的算子
            // 只有这里真正删除了算子，shared_ptr 计数才会归零，防止 bad_weak_ptr
            ops.erase(std::remove_if(ops.begin(), ops.end(), [&](const Operator &o) {
                // 如果算子不是 MatMul，且它的所有输出都没有人用了，就删掉它
                if (o->getOpType() == OpType::MatMul) return false;
                for (auto &out : o->getOutputs()) {
                    if (!out->getTargets().empty()) return false;
                }
                // 删除前，断开它与输入 Tensor 的联系
                for (auto &in : o->getInputs()) in->removeTarget(o);
                return true;
            }), ops.end());

            // 移除中间残留的中间 Tensor 
            tensors.erase(std::remove_if(tensors.begin(), tensors.end(), [](const Tensor &t) {
                // 保留图输入（无 source）、图输出（无 targets）
                // 移除既没来源又没去向的中间变量
                return t->getSource() == nullptr && t->getTargets().empty();
            }), tensors.end());
        }

        // 重建算子间的拓扑连接
        // 因为前面的逻辑打乱了算子间的双向弱引用，必须全部重来
        for (auto &op : ops) {
            op->predecessors.clear();
            op->successors.clear();
        }
        for (auto &op : ops) {
            for (auto &inTensor : op->getInputs()) {
                if (auto sourceOp = inTensor->getSource()) {
                    op->addPredecessors(sourceOp);
                    sourceOp->addSuccessors(op);
                }
            }
        }

        // 重新触发形状推导，确保 MatMul 的 mnk 不再是 0
        this->sorted = false;
    }


    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc() {
        IT_ASSERT(topo_sort() == true);

        std::unordered_map<TensorObj *, size_t> offsets;
        std::unordered_map<TensorObj *, size_t> refCount;

        // 初始化引用计数
        for (auto &tensor : tensors) {
            refCount[tensor.get()] = tensor->getTargets().size();
        }

        // 给没有源头的tensor也要分配空间
        for (auto &tensor : tensors) {
            // 如果一个 Tensor 没有来源算子，说明它是输入或权重
            if (!tensor->getSource()) {
                size_t size = tensor->getBytes();
                if (size > 0) {
                    offsets[tensor.get()] = allocator.alloc(size);
                }
            }
        }
        // =================================================================

        // 离线规划：遍历算子
        for (auto &op : ops) {
            // 分配输出
            for (auto &tensor : op->getOutputs()) {
                size_t size = tensor->getBytes();
                if (size > 0) {
                    offsets[tensor.get()] = allocator.alloc(size);
                }
            }

            // 释放输入
            for (auto &tensor : op->getInputs()) {
                auto t_ptr = tensor.get();
                if (refCount.count(t_ptr)) {
                    refCount[t_ptr]--;
                    if (refCount[t_ptr] == 0) {
                        // 只有分配过的（在 offsets 里的）才需要释放
                        if (offsets.count(t_ptr)) {
                            allocator.free(offsets[t_ptr], tensor->getBytes());
                        }
                    }
                }
            }
        }

        // 拿到物理大内存并绑定
        void *runtime_ptr = allocator.getPtr();
        for (auto &tensor : tensors) {
            auto t_ptr = tensor.get();
            if (tensor->getBytes() > 0 && offsets.count(t_ptr)) {
                size_t offset = offsets[t_ptr];
                uint8_t *real_ptr = (uint8_t *)runtime_ptr + offset;
                tensor->setDataBlob(make_ref<BlobObj>(runtime, real_ptr));
            }
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini