#include "MPI_4_PCFG.h"
#include <chrono>
#include <queue>
#include <condition_variable>
#include <cstring>  // 添加这一行，支持 memset 函数
#include<unistd.h>
#include <sstream>
#include<mpi.h>
using namespace std;
void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    MPI_Generate(priority.front()); // 使用MPI版本
    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

void compute_chunk_range(int total, int rank, int size, int &start, int &end) {
    int base = total / size;
    int extra = total % size;
    if (rank < extra) {
        start = rank * (base + 1);
        end = start + base + 1;
    } else {
        start = extra * (base + 1) + (rank - extra) * base;
        end = start + base;
    }
}

void PriorityQueue::MPI_Generate(PT pt) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CalProb(pt);  // 计算概率

    int local_generated = 0;
    guesses.clear(); // 确保当前进程的猜测空间干净

    if (pt.content.size() == 1) {
        // 单段：直接遍历 ordered_values[i] 
        const segment *seg = nullptr;
        auto &s = pt.content[0];
        if (s.type == 1) seg = &m.letters[m.FindLetter(s)];
        else if (s.type == 2) seg = &m.digits[m.FindDigit(s)];
        else if (s.type == 3) seg = &m.symbols[m.FindSymbol(s)];

        int total = pt.max_indices[0];
        int start_idx, end_idx;
        compute_chunk_range(total, rank, size, start_idx, end_idx);

        for (int i = start_idx; i < end_idx; ++i) {
            guesses.emplace_back(seg->ordered_values[i]);
            ++local_generated;
        }
    } else {
       
        std::ostringstream prefix;
        for (size_t i = 0; i < pt.content.size() - 1; ++i) {
            const auto &seg = pt.content[i];
            int idx = pt.curr_indices[i];
            if (seg.type == 1) prefix << m.letters[m.FindLetter(seg)].ordered_values[idx];
            else if (seg.type == 2) prefix << m.digits[m.FindDigit(seg)].ordered_values[idx];
            else if (seg.type == 3) prefix << m.symbols[m.FindSymbol(seg)].ordered_values[idx];
        }

        const segment *last_seg = nullptr;
        const auto &last = pt.content.back();
        if (last.type == 1) last_seg = &m.letters[m.FindLetter(last)];
        else if (last.type == 2) last_seg = &m.digits[m.FindDigit(last)];
        else if (last.type == 3) last_seg = &m.symbols[m.FindSymbol(last)];

        int total = pt.max_indices.back();
        int start_idx, end_idx;
        compute_chunk_range(total, rank, size, start_idx, end_idx);

        for (int i = start_idx; i < end_idx; ++i) {
            guesses.emplace_back(prefix.str() + last_seg->ordered_values[i]);
            ++local_generated;
        }
    }

    // 汇总总生成数量
    int global_generated = 0;
    MPI_Allreduce(&local_generated, &global_generated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    total_guesses = global_generated;

    //主进程输出结果信息
    if (rank == 0) {
        std::cout << "[MPI_Generate] Total guesses: " << total_guesses << std::endl;
    }
}


 void PriorityQueue::PopNextBatchMPI(int batch_size)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int actual_size = std::min(batch_size, static_cast<int>(priority.size()));
    if (actual_size == 0) return;

    // 先取出batch的PT
    std::vector<PT> batch_pts(priority.begin(), priority.begin() + actual_size);

    // 并行处理这些PT
    ProcessBatchMPI(batch_pts);

    // 生成新PT并插入优先队列
    std::vector<PT> new_pts_all;
    for (auto& pt : batch_pts) {
        auto new_pts = pt.NewPTs();
        for (auto& new_pt : new_pts) {
            CalProb(new_pt);
            new_pts_all.push_back(new_pt);
        }
    }

    // 删除已处理的PT
    priority.erase(priority.begin(), priority.begin() + actual_size);

    // 将新PT按概率从高到低插入队列
    for (auto& new_pt : new_pts_all) {
        auto it = std::find_if(priority.begin(), priority.end(),
                               [&](const PT& elem) { return new_pt.prob > elem.prob; });
        priority.insert(it, new_pt);
    }
}

  void PriorityQueue::ProcessBatchMPI(std::vector<PT>& pt_batch)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int batch_size = pt_batch.size();
    int base_count = batch_size / size;
    int remainder = batch_size % size;

    int start = (rank < remainder) ? rank * (base_count + 1) : remainder * (base_count + 1) + (rank - remainder) * base_count;
    int count = (rank < remainder) ? base_count + 1 : base_count;
    int end = start + count;

    std::vector<std::string> local_guesses;
    int local_guess_count = 0;

    for (int i = start; i < end; ++i) {
        std::vector<std::string> pt_guesses;
        GenerateGuessesForPT(pt_batch[i], pt_guesses);

        local_guesses.insert(local_guesses.end(), pt_guesses.begin(), pt_guesses.end());
        local_guess_count += pt_guesses.size();
    }

    guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());

    int global_guess_count = 0;
    MPI_Allreduce(&local_guess_count, &global_guess_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    total_guesses += global_guess_count;
}


 void PriorityQueue::GenerateGuessesForPT(PT& pt, std::vector<std::string>& output_guesses)
{
    CalProb(pt);

    if (pt.content.size() == 1) {
        segment* seg_ptr = nullptr;
        if (pt.content[0].type == 1)
            seg_ptr = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            seg_ptr = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3)
            seg_ptr = &m.symbols[m.FindSymbol(pt.content[0])];

        for (int i = 0; i < pt.max_indices[0]; ++i)
            output_guesses.push_back(seg_ptr->ordered_values[i]);
    }
    else {
        std::string prefix;
        int idx = 0;
        for (int curr_idx : pt.curr_indices) {
            if (pt.content[idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[idx])].ordered_values[curr_idx];
            else if (pt.content[idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[idx])].ordered_values[curr_idx];
            else if (pt.content[idx].type == 3)
                prefix += m.symbols[m.FindSymbol(pt.content[idx])].ordered_values[curr_idx];
            ++idx;
            if (idx == pt.content.size() - 1) break;
        }

        segment* last_seg = nullptr;
        int last_idx = pt.content.size() - 1;
        if (pt.content[last_idx].type == 1)
            last_seg = &m.letters[m.FindLetter(pt.content[last_idx])];
        else if (pt.content[last_idx].type == 2)
            last_seg = &m.digits[m.FindDigit(pt.content[last_idx])];
        else if (pt.content[last_idx].type == 3)
            last_seg = &m.symbols[m.FindSymbol(pt.content[last_idx])];

        for (int i = 0; i < pt.max_indices[last_idx]; ++i)
            output_guesses.push_back(prefix + last_seg->ordered_values[i]);
    }
}