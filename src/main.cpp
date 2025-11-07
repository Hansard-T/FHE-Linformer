#include <iostream>
#include "FHEController.h"
#include <chrono>
#include <filesystem>

#define GREEN_TEXT "\033[1;32m"
#define RED_TEXT "\033[1;31m"

using namespace std::chrono;
namespace fs = std::filesystem;

enum class Parameters { Generate, Load };

FHEController controller;

Ctxt encoder1();

string input_folder;

string text;

bool verbose = false;
Parameters p = Parameters::Load;
bool security128bits = false;

double read_value(const string &filename) {
    ifstream fin(filename);
    double val;
    if (!(fin >> val)) {
        cerr << "读取 " << filename << " 失败" << endl;
        exit(1);
    }
    return val;
}

void setup_environment(int argc, char *argv[]) {
    string command;

    if (argc < 2) {
        cout << "This is FHE-Linformer, an encrypted text classifier based on Transformer.\n\nUsage: ./FHE-Linformer <text_input> [OPTIONS]\n\nthe following [OPTIONS] are available:\n--verbose: activates verbose mode\n--secure: creates parameters with 128 bits of security. Use only if necessary, as it adds computational overhead \n\nExample:\n./FHE-Linformer --verbose\n";
        exit(0);
    } else {
        if (string(argv[1]) == "--generate_keys")
        {
            if (argc > 2 && string(argv[2]) == "--secure") {
                security128bits = true;
            }

            p = Parameters::Generate;
            return;
        }

        for (int i = 1; i < argc; i++) {
            if (string(argv[i]) == "--verbose") {
                verbose = true;
            }
        }

        // if (verbose) cout << "\nCLIENT-SIDE\nTokenizing..." << endl;
        // command = "python3 ../src/python/ExtractEmbeddings.py";
        // system(command.c_str());
        // command = "python3 ../src/python/extract_parameters_numeric.py";
        // system(command.c_str());
        // command = "python3 ../src/python/dimReduce.py";
        // system(command.c_str());
        // command = "python3 ../src/python/mergeweights.py"
        // system(command.c_str());

    }

}

int main(int argc, char* argv[]) {
    input_folder = "/Users/tangxianning/Downloads/FHE-Linformer/src/tmp_embeddings/";
    setup_environment(argc, argv);

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

    int timing = (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0;
    if (verbose) cout << endl << "The evaluation of the FHE circuit took: " << timing << " seconds." << endl;
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

    
    vector<Ctxt> inputs_E;
    for (int i = 0; i < 32; i++) {
        inputs_E.push_back(controller.read_expanded_input("../input/XE_" + to_string(i) + ".txt")); // 32 * 128
    }

    vector<Ctxt> inputs_F;
    for (int i = 0; i < 32; i++) {
        inputs_F.push_back(controller.read_expanded_input("../input/XF_" + to_string(i) + ".txt")); // 32 * 128
    }

    vector<Ctxt> inputs;
    for (int i = 0; i < inputs_count; i++) {
        inputs.push_back(controller.read_expanded_input(input_folder + "input_" + to_string(i) + ".txt"));
    }

    Ptxt query_w = controller.read_plain_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WQ_weight.txt");
    Ptxt query_b = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WQ_bias.txt");
    Ptxt key_w = controller.read_plain_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WK_weight.txt");
    Ptxt key_b = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WK_bias.txt");

    vector<Ctxt> Q = controller.matmulRE(inputs, query_w, query_b);
    vector<Ctxt> K = controller.matmulRE(inputs_E, key_w, key_b);

    Ctxt K_wrapped = controller.wrapUpRepeated(K);

    Ctxt scores = controller.matmulScores(Q, K_wrapped);
    scores = controller.eval_exp(scores, inputs.size());

    Ctxt scores_sum = controller.rotsum(scores, 128, 128);
    Ctxt scores_denominator = controller.eval_inverse_naive(scores_sum, 2, 5000);

    scores = controller.mult(scores, scores_denominator);

    vector<Ctxt> unwrapped_scores = controller.unwrapScoresExpanded(scores, inputs.size());

    Ptxt value_w = controller.read_plain_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WV_weight.txt");
    Ptxt value_b = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WV_bias.txt");
    

    vector<Ctxt> V = controller.matmulRE(inputs_F, value_w, value_b);
    Ctxt V_wrapped = controller.wrapUpRepeated(V);

    vector<Ctxt> output = controller.matmulRE(unwrapped_scores, V_wrapped, 128, 128);

    if (verbose) cout << "The evaluation of Self-Attention took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print(output[0], 128, "Self-Attention (Repeated)");

    start = high_resolution_clock::now();

    Ptxt dense_w = controller.read_plain_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WO_weight.txt", output[0]->GetLevel());
    Ptxt dense_b = controller.read_plain_expanded_input("../weights-20NG/linformer_transformerLayers_transformer0_selfAttn_WO_bias.txt", output[0]->GetLevel() + 1);

    output = controller.matmulCR(output, dense_w, dense_b);

    for (int i = 0; i < output.size(); i++) {
        output[i] = controller.add(output[i], inputs[i]);
    }

    double c10 = read_value("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine1_c0.txt");
    double c11 = read_value("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine1_c1.txt");
    double c12 = read_value("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine1_c2.txt");

    size_t S = output.size();
    double fL1 = c10 + c11 / sqrt(S) + c12 / S;

    cout << "fL1: " << fL1 << endl;

    vector<Ctxt> output_0;
    vector<Ctxt> output_1;
    for (int i = 0;i < 128; i++) {
        output_0.push_back(output[i]);
    }
    for (int i = 128; i < output.size(); i++) {
        output_1.push_back(output[i]);
    }

    Ctxt wrappedOutput_0 = controller.wrapUpExpanded(output_0);
    Ctxt wrappedOutput_1 = controller.wrapUpExpanded(output_1);

    Ptxt a1 = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine1_a.txt", wrappedOutput_0->GetLevel(), fL1);
    Ptxt b1 = controller.read_plain_repeated_input("../weights-20NG/linformer_transformerLayers_transformer0_ffn_affine1_b.txt", wrappedOutput_0->GetLevel(), fL1);

    wrappedOutput_0 = controller.mult(wrappedOutput_0, a1);
    wrappedOutput_0 = controller.add(wrappedOutput_0, b1);
    wrappedOutput_1 = controller.mult(wrappedOutput_1, a1);
    wrappedOutput_1 = controller.add(wrappedOutput_1, b1);

    wrappedOutput_0 = controller.bootstrap(wrappedOutput_0);
    wrappedOutput_1 = controller.bootstrap(wrappedOutput_1);

    Ctxt output_copy_0 = wrappedOutput_0->Clone();
    Ctxt output_copy_1 = wrappedOutput_1->Clone();

    output_0 = controller.unwrapExpanded(wrappedOutput_0, 128);
    output_1 = controller.unwrapExpanded(wrappedOutput_1, inputs.size() - 128);

    if (verbose) cout << "The evaluation of Self-Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print_expanded(output_0[0], 0, 128, "Self-Output (Expanded)");

    start = high_resolution_clock::now();

    Ptxt intermediate_w_1 = controller.read_plain_input("../weights-20NG/ffn_W1_block_0.txt", wrappedOutput_0->GetLevel());
    Ptxt intermediate_w_2 = controller.read_plain_input("../weights-20NG/ffn_W1_block_1.txt", wrappedOutput_0->GetLevel());
    Ptxt intermediate_w_3 = controller.read_plain_input("../weights-20NG/ffn_W1_block_2.txt", wrappedOutput_0->GetLevel());
    Ptxt intermediate_w_4 = controller.read_plain_input("../weights-20NG/ffn_W1_block_3.txt", wrappedOutput_0->GetLevel());

    vector<Ptxt> dense_weights = {intermediate_w_1, intermediate_w_2, intermediate_w_3, intermediate_w_4};

    Ptxt intermediate_bias = controller.read_plain_input("../weights-20NG/linformer_transformerLayers_transformer0_ffn_Wffn_0_bias.txt", wrappedOutput_0->GetLevel() + 1);

    output_0 = controller.matmulRElarge(output_0, dense_weights, intermediate_bias);
    output_1 = controller.matmulRElarge(output_1, dense_weights, intermediate_bias);

    output_0 = controller.generate_containers(output_0, nullptr);
    output_1 = controller.generate_containers(output_1, nullptr);

    for (int i = 0; i < output_0.size(); i++) {
        controller.print_min_max(output_0[i]);
    }
    for (int i = 0; i < output_1.size(); i++) {
        controller.print_min_max(output_1[i]);
    }

    return 0;
}