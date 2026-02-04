#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

   
    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A = inputs[0];
        auto B = inputs[1];
        auto shapeA = A->getDims();
        auto shapeB = B->getDims();
        int rankA = A->getRank();
        int rankB = B->getRank();

        // 提取 A 的 M 和 K
        int M = transA ? shapeA[rankA - 1] : shapeA[rankA - 2];
        int KA = transA ? shapeA[rankA - 2] : shapeA[rankA - 1];

        // 提取 B 的 N 和 K
        int KB = transB ? shapeB[rankB - 1] : shapeB[rankB - 2];
        int N = transB ? shapeB[rankB - 2] : shapeB[rankB - 1];

        // 校验 K 维度是否匹配 
        IT_ASSERT(KA == KB, "MatMul dimension mismatch on K!");

        // 处理 Batch 维度（即除了最后两维以外的所有维度）
        Shape batchA(shapeA.begin(), shapeA.end() - 2);
        Shape batchB(shapeB.begin(), shapeB.end() - 2);
        
        // 利用广播
        Shape resShape = infer_broadcast(batchA, batchB);

        // 把计算出的 M 和 N 补在最后
        resShape.push_back(M);
        resShape.push_back(N);

        return {{resShape}};
    }

} // namespace infini