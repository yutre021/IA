#include<iostream>
#include<cmath> // For exp() and other mathematical functions

using namespace std;

// Sigmoid function for activation
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));  // Sigmoid formula
}

// Function that calculates the weighted sum and applies the exponential function
float sum(int lx, int ly, int wX, int wY, float sumY, int fuzzle, int a, bool IsNegativeOrNot) {
    float out1 = 0;
    // Calculate the sum of variables
    sumY = lx * wX + ly * wY;
    
    // Check if the exponential is negative or positive
    if (IsNegativeOrNot == false) {
        // Calculate the positive exponential
        out1 = 1 / (1 + (a * exp(sumY - fuzzle)));
        return out1;
    } else {
        // Calculate the negative exponential
        out1 = 1 / (1 + (-a * exp(sumY - fuzzle)));
        return out1;
    }
}

int main() {
    int count = 0;
    float sumY = 0;
    int lX, lY, wX, wY;
    
    do {
        // Receive input values
        cout << "Enter values for lX, lY, wX, wY:" << endl;
        cin >> lX >> lY >> wX >> wY;

        // Conditions based on the value of 'count' to call the 'sum' function
        switch (count) {
            case 0:
                sumY = sum(lX, lY, wX, wY, sumY, 2, 10, false);
                break;
            case 1:
                sumY = sum(lX, lY, wX, wY, sumY, 6, 10, true);
                break;
            case 2:
                sumY = sum(lX, lY, wX, wY, sumY, 8, 10, false);
                break;
            default:
                break;
        }
        count++;
    } while (count < 3);
    
    // Display the final result
    cout << "The final result of sumY is: " << sumY << endl;
    return 0;
}
