#include "neuronet.h"

using namespace nuronet;

double a_function(double d) {
    return d;
}

double b_yes_function(double d) {
    if (d >= 0.5) return 1;
    return 0;
}


double c_function(double d) {
    if (d >= 0.5) return 1;
    return 0;
}

double sigma_function(double d) {
    return 1/(1 + std::exp(-d));
    //return (1 - std::exp(-d));
}


void test_creating() {
    // making layer 1
    Neuron a1(&a_function);
    a1.AddWeight(0.25);
    a1.AddWeight(0.5);

    Neuron a2(&a_function);
    a2.AddWeight(0.25);
    a2.AddWeight(-0.4);

    Neuron a3(&a_function);
    a3.AddWeight(0.0);
    a3.AddWeight(0.9);

    NeuronsLayer layer1;
    layer1.AddNeron(a1);
    layer1.AddNeron(a2);
    layer1.AddNeron(a3);



    // makeing layer2

    NeuronsLayer layer2;
    Neuron b1(&b_yes_function);
    b1.AddWeight(-1);
    Neuron b2(&b_yes_function);
    b2.AddWeight(1);
    layer2.AddNeron(b1);
    layer2.AddNeron(b2);


    //Making layer 3
    NeuronsLayer layer3;
    Neuron c1(&c_function);
    c1.AddWeight(1);
    layer3.AddNeron(c1);

    SkyNet neuronet;
    neuronet.AddLayer(layer1);
    neuronet.AddLayer(layer2);
    neuronet.AddLayer(layer3);

    std::vector<bool> state1 = { false, false, false };
    std::vector<bool> state2 = { false, false, true };
    std::vector<bool> state3 = { false, true, false };
    std::vector<bool> state4 = { false, true, true };
    std::vector<bool> state5 = { true, false, false };
    std::vector<bool> state6 = { true, false, true };
    std::vector<bool> state7 = { true, true, false };
    std::vector<bool> state8 = { true, true, true };


    SkyNet::InputSignals inputs1(state1);


    //no drink, no rain, no friends - don't go to the bar         0 0 0
    
    assert(neuronet.GetDecision(inputs1) == 0);

    //no drink, no rain, friends - go to the bar             0 0 1
    SkyNet::InputSignals inputs2(state2);
    
    assert(neuronet.GetDecision(inputs2) == 1);


    //no drink, rain, no friends - don't go to the bar        0 1 0
    SkyNet::InputSignals inputs3(state3);
    assert(neuronet.GetDecision(inputs3) == 0);

    //no drink, rain, friends - go to the bar            0 1 1
    SkyNet::InputSignals inputs4(state4);
    
    assert(neuronet.GetDecision(inputs4) == 1);

    //drink, no rain, no friends - go to the bar               1 0 0
    SkyNet::InputSignals inputs5(state5);
    
    assert(neuronet.GetDecision(inputs5) == 1);

    //drink, no rain, friends - go to the bar            1 0 1
    SkyNet::InputSignals inputs6(state6);
    
    assert(neuronet.GetDecision(inputs6) == 1);

    //drink, rain, no friends - don't go to the bar    1 1 0
    SkyNet::InputSignals inputs7(state7);
    
    assert(neuronet.GetDecision(inputs7) == 0);

    //drink, rain, friends - don't go to the bar      1 1 1 
    SkyNet::InputSignals inputs8(state8);
    assert(neuronet.GetDecision(inputs8) == 0);




}



void test_learning() {

    // making layer 1
    Neuron a1(&a_function);
    a1.AddWeight(fRand(0,1));
    a1.AddWeight(fRand(0, 1));

    Neuron a2(&a_function);
    a2.AddWeight(fRand(0, 1));
    a2.AddWeight(fRand(0, 1));

    Neuron a3(&a_function);
    a3.AddWeight(fRand(0, 1));
    a3.AddWeight(fRand(0, 1));

    NeuronsLayer layer1;
    layer1.AddNeron(a1);
    layer1.AddNeron(a2);
    layer1.AddNeron(a3);



    // makeing layer2

    NeuronsLayer layer2;
    Neuron b1(&sigma_function);
    b1.AddWeight(fRand(0, 1));

    Neuron b2(&sigma_function);
    b2.AddWeight(fRand(0, 1));

    layer2.AddNeron(b1);
    layer2.AddNeron(b2);


    //Making layer 3
    NeuronsLayer layer3;
    Neuron c1(&sigma_function);
    c1.AddWeight(1);
    layer3.AddNeron(c1);

    SkyNet neuronet;
    neuronet.AddLayer(layer1);
    neuronet.AddLayer(layer2);
    neuronet.AddLayer(layer3);


    std::vector<bool> state1= { false, false, false };
    std::vector<bool> state2 = { false, false, true };
    std::vector<bool> state3 = { false, true, false };
    std::vector<bool> state4 = { false, true, true };
    std::vector<bool> state5 = { true, false, false };
    std::vector<bool> state6 = { true, false, true };
    std::vector<bool> state7 = { true, true, false };
    std::vector<bool> state8 = { true, true, true };

    std::vector<std::string> conditions = {
    "state = { false, false, false }=",
    "state = { false, false, true }=",
    "state = { false, true, false }=",
    "state = { false, true, true }=",
    "state = { true, false, false }=",
    "state = { true, false, true }=",
    "state = { true, true, false }=",
    "state = { true, true, true }=",
    };


    std::vector<SkyNet::InputSignals> set_of_input_state = 
    {
        SkyNet::InputSignals(state1),
        SkyNet::InputSignals(state2),
        SkyNet::InputSignals(state3),
        SkyNet::InputSignals(state4),
        SkyNet::InputSignals(state5),
        SkyNet::InputSignals(state6),
        SkyNet::InputSignals(state7),
        SkyNet::InputSignals(state8),

    };

    std::vector<double> expected_decisions = {
        0, 1, 0, 1, 1, 1, 0, 1
    };


    std::cout << "Before learning:" << std::endl;
    neuronet.PrintDump();
    for (int i = 0; i < set_of_input_state.size(); i++) {
        double result = neuronet.GetDecision(set_of_input_state[i]);
        std::cout << conditions[i] << " Decision = " << result << "; Eroor of learning ="<< result- expected_decisions[i]<< std::endl;

    }


    neuronet.Learning(set_of_input_state, expected_decisions, 6000, 0.08);
    
    std::cout << "After learning:" << std::endl;

    for (int i = 0; i < set_of_input_state.size(); i++) {
        double result = neuronet.GetDecision(set_of_input_state[i]);
        std::cout << conditions[i] << " Decision = " << result << "; Eroor of learning =" << result - expected_decisions[i] << std::endl;

    }
    neuronet.PrintDump();



}


int main(){
    

    test_creating();

    test_learning();

    return 0;
}
