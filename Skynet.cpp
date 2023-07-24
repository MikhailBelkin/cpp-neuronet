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



void test1() {

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

    SkyNet::InputSignals inputs;
    
    //drink, no rain, no friends - go to the bar         1 0 0
    inputs.bool_signals = { true, false, false };          
    assert(neuronet.GetDecision(inputs) == 1);

    //drink, no rain, friends - go to the bar             1 0 1
    inputs.bool_signals = { true, false, true };
    assert(neuronet.GetDecision(inputs) == 1);


    //drink, rain, no friends - don't go to the bar        1 1 0
    inputs.bool_signals = { true, true, false };
    assert(neuronet.GetDecision(inputs) == 0);

    //drink, rain, friends - don't go to the bar            1 1 1
    inputs.bool_signals = { true, true, true };
    assert(neuronet.GetDecision(inputs) == 0);

    //no drink, rain, friends - go to the bar               0 1 1
    inputs.bool_signals = { false, true, true };
    assert(neuronet.GetDecision(inputs) == 1);

    //no drink, no rain, friends - go to the bar            0 0 1
    inputs.bool_signals = { false, false, true };
    assert(neuronet.GetDecision(inputs) == 1);

    //no drink, no rain, no friends - don't go to the bar    0 0 0
    inputs.bool_signals = { false, false, false };
    assert(neuronet.GetDecision(inputs) == 0);

    //no drink, rain, no friends - don't go to the bar      0 1 0 
    inputs.bool_signals = { false, true, false };
    assert(neuronet.GetDecision(inputs) == 0);


}


int main(){
    

    test1();
    return 0;
}
