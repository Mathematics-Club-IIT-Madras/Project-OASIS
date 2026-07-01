#include <bits/stdc++.h>
#include <charconv>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
using namespace chrono;

double string_to_double(string_view sv){
    double num=0;
    auto [ptr,err]=from_chars(sv.data(),sv.data()+sv.size(),num);
    if(err!=errc{}){return 0;}
    return num;
}

vector<double> file_to_vector(string filename){
    ifstream file(filename);
    string line;
    vector<double>data;
    while(getline(file,line)){
        string_view l_view(line);
        size_t left=0;
        size_t right=l_view.find(',');
        left=right+1;
        string_view token=l_view.substr(left);
        data.push_back(string_to_double(token));
    }
    return data;
}


int main(){
    auto start=high_resolution_clock::now();
    vector<double>data=file_to_vector("data.csv");
    Map<VectorXd> vxd(data.data(),data.size());
    double mean=vxd.mean();
    double var=(vxd.array()-vxd.mean()).square().sum()/vxd.size();
    auto end=high_resolution_clock::now();
    duration<double>time=end-start;
    cout<<"time: "<<time.count()<<"\n";
    cout<<"Mean: "<<mean;
    cout<<"\n";
    cout<<"Variance: "<<var;
}

//time: 0.0013s