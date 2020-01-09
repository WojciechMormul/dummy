#include <thread>
#include <future>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <future>

using namespace std;
using namespace std::chrono;

void f1(std::promise<int> * promObj, int value)
{
    std::this_thread::sleep_for(seconds(3));
    promObj->set_value(value);
}

int f2(int value)
{
    std::this_thread::sleep_for(seconds(3));
    return value;
}

static const int N = 3;
//https://solarianprogrammer.com/2011/12/16/cpp-11-thread-tutorial/


auto f = [](int a, int b) {return a+b;};

int main()
{

    /*
    std::thread threads[N];
    std::vector<std::future<int>> futures;
    std::vector<std::promise<int>> promises(N);
    //for (auto& p : promises) futures.push_back(p.get_future());
    for (int i = 0; i < N; ++i){
        futures.push_back(promises[i].get_future());
        threads[i] = std::thread(f1, &promises[i], i);
    }
    for (int i = 0; i < N; ++i){
        std::cout<<futures[i].get()<<std::endl;
        threads[i].join();
    }
    return 0;*/

    /*
    std::vector<std::future<int>> futures;
    for (int i=0;i<N;i++){
       std::future<int> result = std::async(f2, i);
       futures.emplace_back(std::move(result));
    }
    while (!futures.empty()){
        std::future<int> result = std::move(futures.back());
        cout<<result.get()<<endl;
        futures.pop_back();
    }
    return 0;*/


    //int b = f(3,4);
    //cout<<b<<endl;



    std::packaged_task<int(int, int)> task(f);

    std::future<int> result = task.get_future();
    std::thread th(std::move(task), 4, 5);


    cout<<result.get()<<endl;
    th.join();

}
