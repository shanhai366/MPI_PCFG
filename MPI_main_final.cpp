#include "MPI_4_PCFG.h"
#include "md5.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace chrono;

double get_time_in_seconds(system_clock::time_point start, system_clock::time_point end) {
    return duration_cast<microseconds>(end - start).count() / 1e6;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    PriorityQueue q;
    double time_train = 0, time_hash = 0, time_guess = 0;

    if (rank == 0) {
        cout << "Starting model training with " << size << " MPI processes..." << endl;
    }

    auto t_train_start = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto t_train_end = system_clock::now();
    time_train = get_time_in_seconds(t_train_start, t_train_end);

    if (rank == 0) {
        cout << " Model training completed in " << time_train << " seconds.\n" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);  // 所有进程等待训练完成

    //  加载并广播测试数据
    unordered_set<string> test_set;
    vector<string> test_passwords;

    if (rank == 0) {
        ifstream infile("./input/Rockyou-singleLined-full.txt");
        string pw;
        int count = 0;
        while (infile >> pw && count < 1000000) {
            test_passwords.push_back(pw);
            test_set.insert(pw);
            count++;
        }
        cout << " Loaded " << count << " test passwords.\n" << endl;
    }

    int test_size = test_passwords.size();
    MPI_Bcast(&test_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        test_passwords.resize(test_size);
    }

    for (int i = 0; i < test_size; ++i) {
        int len = rank == 0 ? test_passwords[i].length() : 0;
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            test_passwords[i].resize(len);
        }
        MPI_Bcast(&test_passwords[i][0], len, MPI_CHAR, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            test_set.insert(test_passwords[i]);
        }
    }

    //密码生成与哈希验证
    q.init();
    int local_cracked = 0, global_cracked = 0;

    if (rank == 0) {
        cout << "Starting password generation and cracking..." << endl;
    }

    int batch_guess_count = 0, total_history = 0;
    auto t_guess_start = system_clock::now();

    const int hash_threshold = 1000000;
    const int print_interval = 100000;
    const int stop_threshold = 10000000;

    while (!q.priority.empty()) {
        q.PopNext();  // 调用并行生成

        int local_guesses = q.guesses.size();
        int global_guesses = 0;
        MPI_Allreduce(&local_guesses, &global_guesses, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        q.total_guesses = global_guesses;

        batch_guess_count += q.total_guesses;

        if (batch_guess_count >= print_interval) {
            if (rank == 0) {
                cout << "Total guesses generated so far: " << total_history + batch_guess_count << endl;
            }

            if (total_history + batch_guess_count >= stop_threshold) {
                auto t_guess_end = system_clock::now();
                time_guess = get_time_in_seconds(t_guess_start, t_guess_end);

                MPI_Allreduce(&local_cracked, &global_cracked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

                if (rank == 0) {
                    cout << "\n Report" << endl;
                    cout << "Guessing Time  : " << time_guess - time_hash << " sec" << endl;
                    cout << "Hashing Time   : " << time_hash << " sec" << endl;
                    cout << "Training Time  : " << time_train << " sec" << endl;
                    cout << "Cracked Count  : " << global_cracked << endl;
                    cout << "Total Guesses  : " << total_history + batch_guess_count << endl;
                }
                break;
            }
        }

        // 当生成数达到阈值，进行哈希验证
        if (batch_guess_count >= hash_threshold) {
            auto t_hash_start = system_clock::now();

            for (const string& pw : q.guesses) {
                if (test_set.count(pw)) {
                    local_cracked++;
                }
                bit32 state[4];
                MD5Hash(pw, state);
            }

            auto t_hash_end = system_clock::now();
            time_hash += get_time_in_seconds(t_hash_start, t_hash_end);

            MPI_Allreduce(&local_cracked, &global_cracked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            if (rank == 0) {
                cout << "Batch complete. Cracked so far: " << global_cracked << "\n" << endl;
            }

            total_history += batch_guess_count;
            batch_guess_count = 0;
            q.guesses.clear();
        }
    }

    MPI_Allreduce(&local_cracked, &global_cracked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "\nSummary " << endl;
        cout << "MPI Processes     : " << size << endl;
        cout << "Total Cracked     : " << global_cracked << endl;
        cout << "Crack Rate        : " << fixed << setprecision(2)
             << (double)global_cracked / test_size * 100 << " %" << endl;
    }

    MPI_Finalize();
    return 0;
}
