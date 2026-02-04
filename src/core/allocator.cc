#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        
        // 根据之前实现的block来高效分配内存
        size_t bestAddr = (size_t)-1;
        size_t minFragmentSize = (size_t)-1;
        auto bestIt = freeBlocks.end();

        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            size_t blockSize = it->second;
            if (blockSize >= size) {
                // 计算剩下的碎片大小 然后比较谁是最合适的
                size_t fragmentSize = blockSize - size;
                if (fragmentSize < minFragmentSize) {
                    minFragmentSize = fragmentSize;
                    bestAddr = it->first;
                    bestIt = it;
                }
                // 如果正好相等，直接就是最优解，不用找了
                if (fragmentSize == 0) break;
            }
        }

        // 处理分配逻辑
        if (bestIt != freeBlocks.end()) {
            // 在空闲块中找到了位置
            size_t allocatedAddr = bestAddr;
            size_t remainingSize = bestIt->second - size;

            // 从空闲列表中移除旧块
            freeBlocks.erase(bestIt);

            // 如果剩下的空间还够用（大于0），就把剩下的部分重新插入 freeBlocks
            if (remainingSize > 0) {
                freeBlocks[allocatedAddr + size] = remainingSize;
            }
            
            return allocatedAddr;
        } else {
            // 空闲块里没位置了，从末尾扩容 
            size_t allocatedAddr = this->used;
            this->used += size;

            // 更新峰值，Peak 决定了最终模型运行需要申请的物理内存总大小
            if (this->used > this->peak) {
                this->peak = this->used;
            }

            return allocatedAddr;
        }

        return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        // 通过回收时判断连续块的位置和大小 来进行合并 减少内存碎块
        // 先把当前释放的块放进 map
        // insert 返回一个 pair，first 是指向插入位置的迭代器
        auto it = freeBlocks.insert({addr, size}).first;

        // 尝试向后合并 
        // 看看我的结尾是不是正好挨着下一个块的开头
        auto next = std::next(it);
        if (next != freeBlocks.end() && (it->first + it->second == next->first)) {
            it->second += next->second; 
            freeBlocks.erase(next);     
        }

        // 尝试向前合并 
        // 看看我前面的块，它的结尾是不是正好挨着我的开头
        if (it != freeBlocks.begin()) {
            auto prev = std::prev(it);
            if (prev->first + prev->second == it->first) {
                prev->second += it->second; 
                freeBlocks.erase(it);       
            }
        }

        // 尾部收缩逻辑 如果释放完毕了
        if (it->first + it->second == this->used) {
            this->used = it->first; 
            freeBlocks.erase(it);   
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
