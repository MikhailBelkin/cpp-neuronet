#include "neuronet.h"
#include <fstream>


using namespace neuronet;

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
    Neuron<bool> a1(&a_function);
    a1.AddWeight(0.25);
    a1.AddWeight(0.5);

    Neuron<bool> a2(&a_function);
    a2.AddWeight(0.25);
    a2.AddWeight(-0.4);

    Neuron<bool> a3(&a_function);
    a3.AddWeight(0.0);
    a3.AddWeight(0.9);

    NeuronsLayer<bool> layer1;
    layer1.AddNeron(a1);
    layer1.AddNeron(a2);
    layer1.AddNeron(a3);



    // makeing layer2

    NeuronsLayer<bool> layer2;
    Neuron<bool> b1(&b_yes_function);
    b1.AddWeight(-1);
    Neuron<bool> b2(&b_yes_function);
    b2.AddWeight(1);
    layer2.AddNeron(b1);
    layer2.AddNeron(b2);


    //Making layer 3
    NeuronsLayer<bool> layer3;
    Neuron<bool> c1(&c_function);
    c1.AddWeight(1);
    layer3.AddNeron(c1);

    SkyNet<bool> neuronet;
    neuronet.AddLayer(layer1);
    neuronet.AddLayer(layer2);
    neuronet.AddLayer(layer3, true);

    std::vector<bool> state1 = { false, false, false };
    std::vector<bool> state2 = { false, false, true };
    std::vector<bool> state3 = { false, true, false };
    std::vector<bool> state4 = { false, true, true };
    std::vector<bool> state5 = { true, false, false };
    std::vector<bool> state6 = { true, false, true };
    std::vector<bool> state7 = { true, true, false };
    std::vector<bool> state8 = { true, true, true };


    SkyNet<bool>::InputSignals inputs1(state1);


    //no drink, no rain, no friends - don't go to the bar         0 0 0
    
    assert(neuronet.GetDecision(inputs1)[0] == 0);

    //no drink, no rain, friends - go to the bar             0 0 1
    SkyNet<bool>::InputSignals inputs2(state2);
    
    assert(neuronet.GetDecision(inputs2)[0] == 1);


    //no drink, rain, no friends - don't go to the bar        0 1 0
    SkyNet<bool>::InputSignals inputs3(state3);
    assert(neuronet.GetDecision(inputs3)[0] == 0);

    //no drink, rain, friends - go to the bar            0 1 1
    SkyNet<bool>::InputSignals inputs4(state4);
    
    assert(neuronet.GetDecision(inputs4)[0] == 1);

    //drink, no rain, no friends - go to the bar               1 0 0
    SkyNet<bool>::InputSignals inputs5(state5);
    
    assert(neuronet.GetDecision(inputs5)[0] == 1);

    //drink, no rain, friends - go to the bar            1 0 1
    SkyNet<bool>::InputSignals inputs6(state6);
    
    assert(neuronet.GetDecision(inputs6)[0] == 1);

    //drink, rain, no friends - don't go to the bar    1 1 0
    SkyNet<bool>::InputSignals inputs7(state7);
    
    assert(neuronet.GetDecision(inputs7)[0] == 0);

    //drink, rain, friends - don't go to the bar      1 1 1 
    SkyNet<bool>::InputSignals inputs8(state8);
    assert(neuronet.GetDecision(inputs8)[0] == 0);




}



void test_learning() {

    // making layer 1
    Neuron<bool> a1(&a_function);
    a1.AddWeight(fRand(0,1));
    a1.AddWeight(fRand(0, 1));

    Neuron<bool> a2(&a_function);
    a2.AddWeight(fRand(0, 1));
    a2.AddWeight(fRand(0, 1));

    Neuron<bool> a3(&a_function);
    a3.AddWeight(fRand(0, 1));
    a3.AddWeight(fRand(0, 1));

    NeuronsLayer<bool> layer1;
    layer1.AddNeron(a1);
    layer1.AddNeron(a2);
    layer1.AddNeron(a3);



    // makeing layer2

    NeuronsLayer<bool> layer2;
    Neuron<bool> b1(&sigma_function);
    b1.AddWeight(fRand(0, 1));

    Neuron<bool> b2(&sigma_function);
    b2.AddWeight(fRand(0, 1));

    layer2.AddNeron(b1);
    layer2.AddNeron(b2);


    //Making layer 3
    NeuronsLayer<bool> layer3;
    Neuron<bool> c1(&sigma_function);
    c1.AddWeight(1);
    layer3.AddNeron(c1);

    SkyNet<bool> neuronet;
    neuronet.AddLayer(layer1);
    neuronet.AddLayer(layer2);
    neuronet.AddLayer(layer3, true);


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


    std::vector<SkyNet<bool>::InputSignals> set_of_input_state = 
    {
        SkyNet<bool>::InputSignals(state1),
        SkyNet<bool>::InputSignals(state2),
        SkyNet<bool>::InputSignals(state3),
        SkyNet<bool>::InputSignals(state4),
        SkyNet<bool>::InputSignals(state5),
        SkyNet<bool>::InputSignals(state6),
        SkyNet<bool>::InputSignals(state7),
        SkyNet<bool>::InputSignals(state8),

    };

    std::vector<double> expected_decisions = {
        0, 1, 0, 1, 1, 1, 0, 1
    };


    std::cout << "Before learning:" << std::endl;
    neuronet.PrintDump(std::cout);
    for (int i = 0; i < set_of_input_state.size(); i++) {
        double result = neuronet.GetDecision(set_of_input_state[i])[0];
        std::cout << conditions[i] << " Decision = " << result << "; Eroor of learning ="<< result- expected_decisions[i]<< std::endl;

    }


    neuronet.Learning(set_of_input_state, expected_decisions, 2000, 0.1);
    
    std::cout << "After learning:" << std::endl;

    for (int i = 0; i < set_of_input_state.size(); i++) {
        double result = neuronet.GetDecision(set_of_input_state[i])[0];
        std::cout << conditions[i] << " Decision = " << result << "; Eroor of learning =" << result - expected_decisions[i] << std::endl;

    }
    neuronet.PrintDump(std::cout);
    {
        std::ofstream test_out("test_serialization.proto", std::ios::binary);
        neuronet.Serialize(test_out);
    }
    SkyNet<bool> new_net;
    {
        std::ifstream test_in("test_serialization.proto", std::ios::binary);
        if (test_in.fail()) {
            std::cout << "Can't read serialization file" << std::endl;
        }
        new_net.DeSerialize(test_in);
    }
    std::cout << "Deserializing net:" << std::endl;
    for (int i = 0; i < set_of_input_state.size(); i++) {
        double result = new_net.GetDecision(set_of_input_state[i])[0];
        std::cout << conditions[i] << " Decision = " << result << "; Eroor of learning =" << result - expected_decisions[i] << std::endl;

    }

    new_net.PrintDump(std::cout);
}

const size_t DIGITS_IMAGES_COUNT = 600;

struct DigitsImage {
        DigitsImage() {
            image.resize(28);
        }

 
        std::vector<char> image;

};

struct DigitsArray {
    DigitsArray() {
        labels.resize(DIGITS_IMAGES_COUNT);
        images.resize(DIGITS_IMAGES_COUNT);
    }


    std::vector<int> labels;
    std::vector<DigitsImage> images;


};


void read_labels_and_images_db(DigitsArray& da) {

    std::ifstream in_images("train-images.idx3-ubyte ", std::ios::binary);
    std::ifstream in_labels("train-images.idx3-ubyte ", std::ios::binary);

    char digit;
    std::vector<char> buff(28*28);
    std::vector<char> header(16);
    in_images.read(header.data(), 16);
    in_images.read(header.data(), 8);

    for (int i = 0; i < DIGITS_IMAGES_COUNT; i++) {
        in_images.read(buff.data(), 28*28);
        in_labels.read(&digit, 1);
        da.images[i].image = buff;
        da.labels[i] = int(digit);
    }



}


void Test_Digit_Recognition() {

    DigitsArray da;
    read_labels_and_images_db(da);

    //Creating big Neural Net

    std::vector<Neuron<char>> for_layer_1(28*28, Neuron<char>(16));
    std::vector<Neuron<char>> for_layer_2(16, Neuron<char>(16));
    std::vector<Neuron<char>> for_layer_3(16, Neuron<char>(10));
    std::vector<Neuron<char>> for_layer_4(10, Neuron<char>(1));

    NeuronsLayer<char> layer1(28 * 28, for_layer_1);
    NeuronsLayer<char> layer2(16, for_layer_2);
    NeuronsLayer<char> layer3(16, for_layer_3);
    NeuronsLayer<char> layer4(10, for_layer_4);


    SkyNet<char> neuronet;
    neuronet.AddLayer(layer1);
    neuronet.AddLayer(layer2);
    neuronet.AddLayer(layer3);
    neuronet.AddLayer(layer4, true);



    std::vector<SkyNet<char>::InputSignals> set_of_input_state; 

    std::vector<double> expected_decisions;

    for (int i = 0; i < da.images.size(); i++) {
        SkyNet<char>::InputSignals inp_val(da.images[i].image);
        set_of_input_state.push_back(inp_val);
        expected_decisions.push_back(da.labels[i]);

    }
    neuronet.Learning(set_of_input_state, expected_decisions , 2000, 0.1);


    
    std::cout << "After learning:" << std::endl;

    for (int i = 0; i < set_of_input_state.size(); i++) {
        auto result = neuronet.GetDecision(set_of_input_state[i]);
        std::cout << "Input symbol = " << da.labels[i] << " Result";
    }

    std::ofstream digit_out("digit_serialization.proto", std::ios::binary);
    neuronet.Serialize(digit_out);
    return;
}







int main(){
    

    //test_creating();

    //test_learning();

    Test_Digit_Recognition();

    return 0;
}
