#include "Pythia8/Pythia.h"
#include <fstream>
#include <iostream>

using namespace Pythia8;

int main() {
    // Output file
    std::ofstream outfile("/home/sawini-jana/Documents/particles_172.5.txt");

    // Initialize Pythia
    Pythia pythia;

    // e+ e- collision
    pythia.readString("Beams:idA = 11");   // e-
    pythia.readString("Beams:idB = -11");  // e+
    pythia.readString("Beams:eCM = 2000."); // Q = 2 TeV
    
    pythia.readString("PartonLevel:ISR = off");
    pythia.readString("PartonLevel:FSR = on");
    pythia.readString("HadronLevel:all = on");

    // Enable e+e- → γ*/Z
    pythia.readString("WeakSingleBoson:ffbar2gmZ = on");

    // Force Z/γ* → t tbar
    pythia.readString("23:onIfMatch = 6 -6");

    // Force hadronic decays of W
    pythia.readString("24:onMode = off");
    pythia.readString("24:onIfAny = 1 2 3 4 5"); // u,d,s,c,b

    // Set top mass (we'll generalize later)
    pythia.readString("6:m0 = 172.0");

    // Initialize
    pythia.init();

    int nEvents = 10000;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;

        // Loop over particles
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            if (!pythia.event[i].isVisible()) continue;

            double px = pythia.event[i].px();
            double py = pythia.event[i].py();
            double pz = pythia.event[i].pz();
            double E  = pythia.event[i].e();

            // Save as: event_id px py pz E
            outfile << iEvent << " "<< px << " "<< py << " "<< pz << " " << E  << "\n";
        }
    }

    //pythia.stat();
    outfile.close();

    std::cout << "Event generation complete.\n";
    return 0;
}
