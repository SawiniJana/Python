#include <fastjet/ClusterSequence.hh>
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace fastjet;

double angle_3d(const PseudoJet &a, const PseudoJet &b) {
    double dot = a.px()*b.px() + a.py()*b.py() + a.pz()*b.pz(); //dot product of a and b
    double mag1 = sqrt(a.px()*a.px() + a.py()*a.py() + a.pz()*a.pz()); //magnitude of vector a
    double mag2 = sqrt(b.px()*b.px() + b.py()*b.py() + b.pz()*b.pz()); //magnitude of vector b
    double c = dot / (mag1 * mag2); //computing the cos of angle
    c = max(-1.0, min(1.0, c)); //just in case the cos(angle) goes over [-1,1]
    return acos(c); //returning theta
}

int main() {
    // Step 1: Create some particles (px, py, pz, E)
    double particle_mass[21] = {170.0,170.5,171.0,171.5,172.0,172.5,173.0,173.5,174.0,174.5,175.0,175.5,
    176.0, 176.5, 177.0,177.5,178.0,178.5,179.0,179.5,180.0};
    std::ostringstream ss1, ss2;

    ss2 <<"/home/sawini-jana/hep/EEEC_1000samples.csv";
    ofstream ml_data(ss2.str());
    ml_data << "z1,z2,z3,weight,mt\n";

    for (int txt = 0; txt < 21; txt++){

        ss1.str("");
        ss1.clear();

        ss1 <<"/home/sawini-jana/Documents/particles_"
            << std::fixed << std::setprecision(1)
            <<particle_mass[txt]
            << ".txt";
            
        ifstream MyReadFile(ss1.str());
    
        if (!MyReadFile) {
        std::cerr << "Could not open file: " << ss1.str() << '\n';
        continue;
        }

        map<int, vector<PseudoJet>> events;
        //events is a map storing int keys and vectors as values

        //store the important properties
        double px, py, pz, E;
        int eventid;
        double mt = particle_mass[txt]; //Monte Carlo top mass
        
        //do an event by event clustering 
        int last_event;
        int current_event = -1;

        while (MyReadFile >> eventid >> px >> py >> pz >> E) {
            events[eventid].push_back(PseudoJet(px, py, pz, E)); //events storing details by eventid
        }
        MyReadFile.close();

        cout << "Particles read: " << events.size() << endl; //Number of events

        // Step 2: Define the jet algorithm
        double R = 1.2;
        JetDefinition antikt_def(antikt_algorithm, R);
        //Step 5: Recluster each jet using Cambridge-Aachen
        double Rp = 0.1;
        JetDefinition ca_def(cambridge_algorithm, Rp);

        //a check so that theta stays within [0,2*pi]
        double theta_min = 0.0;
        double theta_max = M_PI;    

        int jet_count = 0;
        //Loop over events
        for (auto &ev : events){
            vector<PseudoJet> &particles = ev.second;
            //particles contain the 4-vectors for each event

            //Cluster anti-kt jets
            ClusterSequence cs(particles, antikt_def);
            //Create Jets
            vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(10.0));

            for (auto &jet : jets){
                //cout<<"Particles in anti-kt jets:"<<jet.constituents().size()<<endl;
                //Recuster using C/A
                vector<PseudoJet> constituents = jet.constituents();
                ClusterSequence cs_ca(constituents, ca_def);
                vector<PseudoJet> subjets = cs_ca.inclusive_jets(10.0);

                //A cap to the minimum energy to not explode the storage and exclude soft particles
                double Emin = 1.0; 
                subjets.erase( 
                    std::remove_if(subjets.begin(), subjets.end(),
                    [Emin](const PseudoJet &sj) { 
                        return sj.E() <= Emin; 
                    }), 
                    subjets.end() 
                );

                if(subjets.size() < 3) continue; //skipping these jets since they wont form an EEEC observable
                jet_count++;

                //cout <<subjets.size() <<endl;
                //for (auto &subjet : subjets){
                //    cout<<"Particles in reclustered jets:"<<subjet.constituents().size()<<endl;
                //}
                int c = 1;
                //Loop over triplets
                for(size_t i = 0; i<subjets.size(); i++){
                    for(size_t j=i+1; j<subjets.size(); j++){
                        for(size_t k=j+1; k<subjets.size(); k++){
                            double Eprod = subjets[i].E() * subjets[j].E() * subjets[k].E();
                            double Q3 = pow(2000.0, 3);

                            if(Eprod/Q3 < pow(10, -6)) continue;
                            //cout<<"Yes DONE!"<<endl;                        
                            
                            double z12 = angle_3d(subjets[i], subjets[j]);
                            double t12 = (1.0 - cos(z12))/2.0; 

                            double z13 = angle_3d(subjets[i], subjets[k]);
                            double t13 = (1.0 - cos(z13))/2.0;

                            double z23 = angle_3d(subjets[j], subjets[k]);
                            double t23 = (1.0 - cos(z13))/2.0;

                            vector<double> zetas = {t12, t13, t23};
                            sort(zetas.begin(), zetas.end()); //sorting the zeta values as suggested by paper

                            double z1 = zetas[0];
                            double z2 = zetas[1];
                            double z3 = zetas[2];
                            
                            ml_data << zetas[0] << "," << zetas[1] << "," << zetas[2] << "," 
                            << Eprod/Q3 << "," << mt << "\n"; //storing in the paper
                            c+= 1;

                        }
                    }
                }
                if (jet_count > 1000){
                    cout<<"HELLO"<<endl;
                    break;}
                
                //cout<<"No of observables per jet: "<< c << endl;
                }
            if (jet_count > 1000) break;
            }

    cout << "Number of jets: " << jet_count << endl;

    }

    return 0;
}
