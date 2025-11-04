#include <iostream>
#include "FHEController.h"
#include <chrono>

#define GREEN_TEXT "\033[1;32m"
#define RED_TEXT "\033[1;31m"

using namespace std::chrono;

enum class Parameters { Generate, Load };

FHEController controller;

vector<Ctxt> encoder1();
Ctxt encoder2(vector<Ctxt> input);
Ctxt pooler(Ctxt input);
Ctxt classifier(Ctxt input);

string input_folder;

string text;

bool verbose = false;
Parameters p = Parameters::Load;
bool security128bits = false;

int main(int argc, char* argv[]) {
    if (p == Parameters::Generate) {
        system("mkdir -p ../keys");
        controller.generate_context(true, security128bits);
        vector<int> rotations = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, -1, -2, -4, -8 -16, -32, -64};
        controller.generate_bootstrapping_and_rotation_keys(rotations, 16384, true, "rotation_keys.txt");
        return 0;
    } else if (p == Parameters::Load) {
        controller.load_context(false);
        controller.load_bootstrapping_and_rotation_keys("rotation_keys.txt", 16384, false);
    }

    system("mkdir -p ../checkpoint");

    if(verbose) cout << "\nSERVER-SIDE\nThe evaluation of the circuit started." << endl;

    auto start = high_resolution_clock::now();

    if(input_folder.empty()) {
        cerr << "The input folder \"" << input_folder << "\" is empty!";
        exit(1);
    }

    Ctxt encoder1output;

    encoder1output = encoder1();
    encoder1output = controller.load_ciphertext("../checkpoint/encoder1output.bin");

    Ctxt pooled = pooler(encoder1output);
    pooled = controller.load_ciphertext("../checkpoint/pooled.bin");

    Ctxt classified = classifier(pooled);

    if (verbose) cout << "The circuit has been evaluated, the results are sent back to the client" << endl << endl;
    if (verbose) cout << "CLIENT-SIDE" << endl;

    if (verbose)
        controller.print(classified, 20, "Output logits");

    vector<double> plain_result = controller.decrypt_tovector(classified, 20);
    
    int timing = (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0;
    if (verbose) cout << endl << "The evaluation of the FHE circuit took: " << timing << " seconds." << endl;

    system(("python3 /Users/tangxianning/Downloads/FHE-Linformer/src/python/compute_20ng_linformer_forward.py"));

}

Ctxt encoder1() {
    auto start = high_resolution_clock::now();

    int inputs_count = 0;

    std::filesystem::path p1 { input_folder };

    for (__attribute__((unused)) auto& p : std::filesystem::directory_iterator(p1))
    {
        ++inputs_count;
    }

    if (verbose) cout << inputs_count << " inputs found!" << endl << endl;

    vector<Ctxt> inputs;
    for (int i = 0; i < inputs_count; i++) {
        inputs.push_back(controller.read_expanded_input(input_folder + "input_" + to_string(i) + ".txt"));
    }

}