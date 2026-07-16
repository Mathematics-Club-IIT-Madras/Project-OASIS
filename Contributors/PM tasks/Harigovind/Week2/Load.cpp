#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

double mean(const std::vector<double>& values)
{
    double sum=0;
    for (int i=0;i<values.size();i++){
        sum+=values[i];
    }
    
    return sum/values.size();
}

double minimum(const std::vector<double>& values){
    
    double minValue=values[0];

    for (int i=1;i<values.size();i++){
        if (values[i]<minValue){
            minValue=values[i];
        }
    }
    return minValue;
}

double maximum(const std::vector<double>& values)
{
    double maxValue = values[0];

    for (int i = 1; i < values.size(); i++)
    {
        if (values[i] > maxValue)
        {
            maxValue = values[i];
        }
    }

    return maxValue;
}

double variance(const std::vector<double>& values){

    double avg=mean(values);
    double sum=0;

    for (int i=0;i<values.size();i++){
        double difference=values[i]-avg;
        sum+=difference*difference;
    }
    return sum/values.size();
}

double standardDeviation(const std::vector<double>& values){
    return std::sqrt(variance(values));
}

int main(){
    std::ifstream file("TravelTime_387.csv");

    if(file.is_open()){
        std::cout<<"File opened succesfully"<<std::endl;
    }
    else{
        std::cout<<"Could not open file"<<std::endl;
        return 1;
    }

    std::string line;
    std::getline(file,line); // To read and ignore the header

    std::vector<double>values;

    while (std::getline(file,line)){
        std::stringstream ss(line);

        std::string timestamp;
        std::string value;

        std::getline(ss,timestamp,',');
        std::getline(ss,value);

        double number=std::stod(value);
        values.push_back(number);

    }
    file.close();

    std::cout<<"Number of values="<<values.size()<<std::endl;
    std::cout<<"Mean ="<<mean(values)<<std::endl;
    std::cout<<"Minimum ="<<minimum(values)<<std::endl;
    std::cout<<"Maximum ="<<maximum(values)<<std::endl;
    std::cout<<"Variance ="<<variance(values)<<std::endl;
    std::cout<<"Standard Deviation ="<<standardDeviation(values)<<std::endl;

    return 0;
}