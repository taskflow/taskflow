/* ***************************************************************************
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */
/* Authors: 
 *   Nicolo' Tonci
 *   Massimo Torquati
 */

#ifndef FF_DGROUPS_H
#define FF_DGROUPS_H

#include <signal.h>
#include <getopt.h>

#include <string>
#include <map>
#include <set>
#include <vector>
#include <sstream>
#include <algorithm>

#include <ff/ff.hpp>

#include <ff/distributed/ff_dprinter.hpp>
#include <ff/distributed/ff_dutils.hpp>
#include <ff/distributed/ff_dintermediate.hpp>
#include <ff/distributed/ff_dgroup.hpp>

#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#ifdef DFF_MPI
#include <mpi.h>
#endif

namespace ff {

class dGroups {
public:
    friend struct GroupInterface;
    static dGroups* Instance(){
      static dGroups dg;
      return &dg;
    }
    
	/*~dGroups() {
		for (auto g : this->groups)
            if (g.second)
				delete g.second;
		groups.clear();
	}*/
	
    Proto usedProtocol;

    void parseConfig(std::string configFile){
      std::ifstream is(configFile);

        if (!is) throw FF_Exception("Unable to open configuration file for the program!");

        try {
            cereal::JSONInputArchive ari(is);
            ari(cereal::make_nvp("groups", this->parsedGroups));
            
            // get the protocol to be used from the configuration file
            try {
                std::string tmpProtocol;
                ari(cereal::make_nvp("protocol", tmpProtocol));
                if (tmpProtocol == "MPI"){
                    #ifdef DFF_MPI
                        this->usedProtocol = Proto::MPI;
                    #else
                        std::cout << "NO MPI support! Falling back to TCP\n";
                        this->usedProtocol = Proto::TCP;
                    #endif 

                } else this->usedProtocol = Proto::TCP;
            } catch (cereal::Exception&) {
                ari.setNextName(nullptr);
                this->usedProtocol = Proto::TCP;
            }

        } catch (const cereal::Exception& e){
            std::cerr << "Error parsing the JSON config file. Check syntax and structure of the file and retry!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void annotateGroup(std::string name, ff_node* parentBB){
      if (annotatedGroups.contains(name)){
        std::cerr << "Group " << name << " created twice. Error!\n"; abort();
      }
      annotatedGroups[name].parentBB = parentBB;
    }

    int size(){ return annotatedGroups.size();}

    void setRunningGroup(std::string g){this->runningGroup = g;}
    
    void setRunningGroupByRank(int rank){
        this->runningGroup = parsedGroups[rank].name;
    }

    /*
    * Set the thread mapping if specified in the configuration file. Otherwise use the default mapping specified in the legacy FastFlow config.hpp file.
    * In config file the mapping can be specified for each group through the key "threadMapping"
    */
    void setThreadMapping(){
      auto g = std::find_if(parsedGroups.begin(), parsedGroups.end(), [this](auto& g){return g.name == this->runningGroup;});
      if (g != parsedGroups.end() && !g->threadMapping.empty())
        threadMapper::instance()->setMappingList(g->threadMapping.c_str());
    }

	  const std::string& getRunningGroup() const { return runningGroup; }

    void forceProtocol(Proto p){this->usedProtocol = p;}
	
    int run_and_wait_end(ff_pipeline* parent){
        if (annotatedGroups.find(runningGroup) == annotatedGroups.end()){
            ff::error("The group %s is not found nor implemented!\n", runningGroup.c_str());
            return -1;
        }

      bool allDeriveFromParent = true; 
      for(auto& [name, ir]: annotatedGroups) 
        if (ir.parentBB != parent) { allDeriveFromParent = false; break; }

      if (allDeriveFromParent) {
        ff_pipeline *mypipe = new ff_pipeline;
        mypipe->add_stage(parent);
        parent = mypipe;
      }

      // qui dovrei creare la rappresentazione intermedia di tutti
      this->prepareIR(parent);

#ifdef PRINT_IR
      this->annotatedGroups[this->runningGroup].print();
#endif

      // buildare il farm dalla rappresentazione intermedia del gruppo che devo rannare
      dGroup _grp(this->annotatedGroups[this->runningGroup]);
      // rannere il farm come sotto!
    if (_grp.run() < 0){
      std::cerr << "Error running the group!" << std::endl;
      return -1;
    }

      if (_grp.wait() < 0){
        std::cerr << "Error waiting the group!" << std::endl;
        return -1;
      }

      #ifdef DFF_MPI
        if (usedProtocol == Proto::MPI)
          if (MPI_Finalize() != MPI_SUCCESS) abort();
      #endif 
        
        return 0;
    }
protected:
    dGroups() : runningGroup() {
        // costruttore
    }
    std::map<ff_node*, std::string> annotated;
private:
    inline static dGroups* i = nullptr;
    std::map<std::string, ff_IR> annotatedGroups;
    
    std::string runningGroup;

    // helper class to parse config file Json
    struct G {
        std::string name;
        std::string address;
        std::string threadMapping;
        int port;
        int batchSize          = DEFAULT_BATCH_SIZE;
        int internalMessageOTF = DEFAULT_INTERNALMSG_OTF;
        int messageOTF         = DEFAULT_MESSAGE_OTF;

        template <class Archive>
        void load( Archive & ar ){
            ar(cereal::make_nvp("name", name));
            
            try {
                std::string endpoint;
                ar(cereal::make_nvp("endpoint", endpoint)); std::vector endp(split(endpoint, ':'));
                address = endp[0]; port = std::stoi(endp[1]);
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}

            try {
                ar(cereal::make_nvp("batchSize", batchSize));
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}

             try {
                ar(cereal::make_nvp("internalMessageOTF", internalMessageOTF));
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}

             try {
                ar(cereal::make_nvp("messageOTF", messageOTF));
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}

            try {
                ar(cereal::make_nvp("threadMapping", threadMapping));
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}

        }
    };

    std::vector<G> parsedGroups;

    static inline std::vector<std::string> split (const std::string &s, char delim) {
        std::vector<std::string> result;
        std::stringstream ss (s);
        std::string item;

        while (getline (ss, item, delim))
            result.push_back (item);

        return result;
    }


    void prepareIR(ff_pipeline* parentPipe){
      ff::ff_IR& runningGroup_IR = annotatedGroups[this->runningGroup];
      runningGroup_IR.protocol = this->usedProtocol; // set the protocol
      ff_node* previousStage = getPreviousStage(parentPipe, runningGroup_IR.parentBB);
      ff_node* nextStage = getNextStage(parentPipe, runningGroup_IR.parentBB);
      // TODO: check coverage all 1st level

      for(size_t i = 0; i < parsedGroups.size(); i++){
        const G & g = parsedGroups[i];

        // throw an error if a group in the configuration has not been annotated in the current program
        if (!annotatedGroups.contains(g.name)) throw FF_Exception("present in the configuration file has not been implemented! :(");
        auto endpoint = this->usedProtocol == Proto::TCP ? ff_endpoint(g.address, g.port) : ff_endpoint(i);
        endpoint.groupName = g.name;

        // annotate the listen endpoint for the specified group
        annotatedGroups[g.name].listenEndpoint = endpoint;
        // set the batch size for each group
        if (g.batchSize > 1) annotatedGroups[g.name].outBatchSize = g.batchSize;
        if (g.messageOTF) annotatedGroups[g.name].messageOTF = g.messageOTF;
        if (g.internalMessageOTF) annotatedGroups[g.name].internalMessageOTF = g.internalMessageOTF;
      }

      // TODO check first level pipeline before strting building the groups.

      // build the map parentBB -> <groups names>
      std::map<ff_node*, std::set<std::string>> parentBB2GroupsName;
      for(auto& p: annotatedGroups) parentBB2GroupsName[p.second.parentBB].insert(p.first);

      // iterate over the 1st level building blocks and the list of create groups 
      for(const auto& pair : parentBB2GroupsName){

        //just build the current previous the current and the next stage of the parentbuilding block i'm going to exeecute  //// TODO: reprhase this comment!
        //if (!(pair.first == previousStage || pair.first == runningGroup_IR.parentBB || pair.first == nextStage))
        //  continue;

        bool isSrc = isSource(pair.first, parentPipe);
        bool isSnk = isSink(pair.first, parentPipe);
        // check if from the under analysis 1st level building block it has been created just one group
        if (pair.second.size() == 1){
          // if the unique group is not the one i'm going to run just skip this 1st level building block
          if ((isSrc || pair.first->isDeserializable()) && (isSnk || pair.first->isSerializable())){
            auto& ir_ = annotatedGroups[*pair.second.begin()];
            ir_.insertInList(std::make_pair(pair.first, SetEnum::L), true);
            ir_.isSink = isSnk; ir_.isSource = isSrc;
          }
          continue; // skip anyway
        }
        
        // if i'm here it means that from this 1st level building block, multiple groups have been created! (An Error or an A2A or a Pipe BB)


        // multiple groups created from a pipeline!
        if (pair.first->isPipe()){
          bool isMainPipe = pair.first == parentPipe;
          ff_pipeline* originalPipe = reinterpret_cast<ff_pipeline*>(pair.first);
          
          // if the pipe coincide with the main Pipe, just ignore the fact that is wrapped around, since it will be handled later on!
          bool iswrappedAround = isMainPipe ? false : originalPipe->isset_wraparound();

          // check that all stages were annotated
          for(ff_node* child : originalPipe->getStages())
            if (!annotated.contains(child)){
              error("When create a group from a pipeline, all the stages must be annotated on a group!");
              abort();
            }

           for(auto& gName: pair.second){
             ff_pipeline* mypipe = new ff_pipeline;

              for(ff_node* child : originalPipe->getStages()){
                if (annotated[child] == gName){
                  // if the current builiding pipe has a stages, and the last one is not the previous i'm currently including there is a problem!
                  if (mypipe->getStages().size() != 0 && originalPipe->get_stageindex(mypipe->get_laststage())+1 != originalPipe->get_stageindex(child)) {
                    error("There are some stages missing in the annottation!\n");
                    abort();
                  } else 
                    mypipe->add_stage(child);
                } 
              }

              bool head = mypipe->get_firststage() == originalPipe->get_firststage();
              bool tail = mypipe->get_laststage() == originalPipe->get_laststage();
             
              if (((head && isSrc && !iswrappedAround) || mypipe->isDeserializable()) && ((tail && isSnk && !iswrappedAround) || mypipe->isSerializable()))  
                annotatedGroups[gName].insertInList(std::make_pair(mypipe, SetEnum::L));
              else {
                error("The group cannot serialize something!\n");
                abort();
              }

              if (head && isSrc) annotatedGroups[gName].isSource = true;
              if (tail && isSnk) annotatedGroups[gName].isSink = true;
            

              if (!tail)
                annotatedGroups[gName].destinationEndpoints.push_back({ChannelType::FWD, annotatedGroups[annotated[originalPipe->get_nextstage(mypipe->get_laststage())]].listenEndpoint});
              else if (iswrappedAround)
                // if this is the last stage & the pipe is wrapper around i connect tot he "head" groups of this pipeline
                annotatedGroups[gName].destinationEndpoints.push_back({ChannelType::FBK, annotatedGroups[annotated[originalPipe->get_firststage()]].listenEndpoint});
             
              // add a new expected connection if i'm not the head or i'm the head and the pipeline is wrapped around
              if (!head || iswrappedAround)
                annotatedGroups[gName].expectedEOS = 1;
    
           }

        } else { // multiple groups created from an all2all!

        std::set<std::pair<ff_node*, SetEnum>> children = getChildBB(pair.first);

        std::erase_if(children, [&](auto& p){
            if (!annotated.contains(p.first)) return false;
            std::string& groupName = annotated[p.first]; 
            if (((isSrc && p.second == SetEnum::L) || p.first->isDeserializable()) && (isSnk || p.first->isSerializable())){
              annotatedGroups[groupName].insertInList(std::make_pair(p.first, p.second)); return true;
            }
            return false;
        });

        if (!children.empty()){
          // seconda passata per verificare se ce da qualche parte c'è copertura completa altrimenti errore
          for(const std::string& gName : pair.second){
            ff_IR& ir = annotatedGroups[gName];
            ir.computeCoverage();
            if (ir.coverageL)
              std::erase_if(children, [&](auto& p){
                if (p.second == SetEnum::R && (isSnk || p.first->isSerializable()) && p.first->isDeserializable()){
                  ir.insertInList(std::make_pair(p.first, SetEnum::R)); return true;
                }
                return false;
              });

            if (ir.coverageR)
              std::erase_if(children, [&](auto& p){
                if (p.second == SetEnum::L && (isSrc || p.first->isDeserializable()) && p.first->isSerializable()){
                  ir.insertInList(std::make_pair(p.first, SetEnum::L)); return true;
                }
                return false;
              });
          }

          // ancora dei building block figli non aggiunti a nessun gruppo, lancio errore e abortisco l'esecuzione
          if (!children.empty()){
            std::cerr << "Some building block has not been annotated and no coverage found! You missed something. Aborting now" << std::endl;
            abort();
          }
        } else  
          for(const std::string& gName : pair.second) {
           annotatedGroups[gName].computeCoverage();
            // compute the coverage anyway
          }

        for(const std::string& _gName : pair.second){
          auto& _ir = annotatedGroups[_gName];
           // set the isSrc and isSink fields
          _ir.isSink = isSnk; _ir.isSource = isSrc;
           // populate the set with the names of other groups created from this 1st level BB
          _ir.otherGroupsFromSameParentBB = pair.second;
        }
        
        }
        //runningGroup_IR.isSink = isSnk; runningGroup_IR.isSource = isSrc;
       
        //runningGroup_IR.otherGroupsFromSameParentBB = pair.second;

      }


      // compute routing for horizzontally splitted A2A
      // this is meaningful only if the group is horizontal and made of an a2a
      if (!runningGroup_IR.isVertical()){
		    assert(runningGroup_IR.parentBB->isAll2All());
        ff_a2a* parentA2A = reinterpret_cast<ff_a2a*>(runningGroup_IR.parentBB);
        {
          ff::svector<ff_node*> inputs;
          for(ff_node* child : parentA2A->getSecondSet()) child->get_in_nodes(inputs);
          runningGroup_IR.rightTotalInputs = inputs.size();
        }
        {
          ff::svector<ff_node*> outputs;
          for(ff_node* child : parentA2A->getFirstSet()) child->get_out_nodes(outputs);
          runningGroup_IR.leftTotalOuputs = outputs.size();
        }
      }

      
      //############# compute the number of excpected input connections
      
      // handle horizontal groups 
      if (runningGroup_IR.hasRightChildren()){
        auto& currentGroups = parentBB2GroupsName[runningGroup_IR.parentBB];
        runningGroup_IR.expectedEOS = std::count_if(currentGroups.cbegin(), currentGroups.cend(), [&](auto& gName){return (!annotatedGroups[gName].isVertical() || annotatedGroups[gName].hasLeftChildren());});
        // if the current group is horizontal count out itsleft from the all horizontals
        if (!runningGroup_IR.isVertical()) runningGroup_IR.expectedEOS -= 1;
      }

      // if the previousStage exists, count all the ouput groups pointing to the one i'm going to run
      // if the runningGroup comes from a pipe, make sure i'm the head 
      if (previousStage && runningGroup_IR.hasLeftChildren() && (!runningGroup_IR.parentBB->isPipe() || inputGroups(parentBB2GroupsName[runningGroup_IR.parentBB]).contains(runningGroup)))
        runningGroup_IR.expectedEOS += outputGroups(parentBB2GroupsName[previousStage]).size();
      
      // FEEDBACK RELATED (wrap around of the main pipe!)
      // if the main pipe is wrapped-around i take all the outputgroups of the last stage of the pipeline and the cardinality must set to the expected input connections
      if (!previousStage && parentPipe->isset_wraparound())
        runningGroup_IR.expectedEOS += outputGroups(parentBB2GroupsName[parentPipe->getStages().back()]).size();


      if (runningGroup_IR.expectedEOS > 0) runningGroup_IR.hasReceiver = true;

      //############ compute the name of the outgoing connection groups
      if (runningGroup_IR.parentBB->isAll2All() && runningGroup_IR.isVertical() && runningGroup_IR.hasLeftChildren() && !runningGroup_IR.wholeParent){
          // inserisci tutte i gruppi di questo bb a destra
          for(const auto& gName: parentBB2GroupsName[runningGroup_IR.parentBB])
            if (!annotatedGroups[gName].isVertical() || annotatedGroups[gName].hasRightChildren())
              runningGroup_IR.destinationEndpoints.push_back({ChannelType::FWD, annotatedGroups[gName].listenEndpoint});
      } 
      else if (runningGroup_IR.parentBB->isPipe() && runningGroup_IR.parentBB != parentPipe){
        if (nextStage && outputGroups(parentBB2GroupsName[runningGroup_IR.parentBB]).contains(runningGroup))
          for(const auto& gName : inputGroups(parentBB2GroupsName[nextStage]))
            runningGroup_IR.destinationEndpoints.push_back({ChannelType::FWD, annotatedGroups[gName].listenEndpoint});
      }
      else {
        if (!runningGroup_IR.isVertical()){
          // inserisci tutti i gruppi come sopra
          for(const auto& gName: parentBB2GroupsName[runningGroup_IR.parentBB])
            if ((!annotatedGroups[gName].isVertical() || annotatedGroups[gName].hasRightChildren()) && gName != runningGroup)
              runningGroup_IR.destinationEndpoints.push_back({ChannelType::INT, annotatedGroups[gName].listenEndpoint});
        }

        if (nextStage)
          for(const auto& gName : inputGroups(parentBB2GroupsName[nextStage]))
            runningGroup_IR.destinationEndpoints.push_back({ChannelType::FWD, annotatedGroups[gName].listenEndpoint});
        else
          // FEEDBACK RELATED (wrap around of the main pipe!)
          if (parentPipe->isset_wraparound()){
            for(const auto& gName : inputGroups(parentBB2GroupsName[parentPipe->getStages().front()]))
              runningGroup_IR.destinationEndpoints.push_back({ChannelType::FBK, annotatedGroups[gName].listenEndpoint});
          }
      }
      
      
      runningGroup_IR.buildIndexes();

      if (!runningGroup_IR.destinationEndpoints.empty()) runningGroup_IR.hasSender = true;
      
      // experimental building the expected routing table for the running group offline (i.e., statically)
      if (runningGroup_IR.hasSender)
        for(auto& [ct, ep] : runningGroup_IR.destinationEndpoints){
            auto& destIR = annotatedGroups[ep.groupName];
            destIR.buildIndexes();
            bool internalConnection = ct == ChannelType::INT || (ct == ChannelType::FWD && runningGroup_IR.parentBB == destIR.parentBB); //runningGroup_IR.parentBB == destIR.parentBB;
            runningGroup_IR.routingTable[ep.groupName] = std::make_pair(destIR.getInputIndexes(internalConnection), ct);
        }

    }

  std::set<std::string> outputGroups(std::set<std::string> groupNames){
    if (groupNames.size() > 1)
      std::erase_if(groupNames, [this](const auto& gName){
        ff_IR& ir = annotatedGroups[gName];
        return ((ir.parentBB->isAll2All() && ir.isVertical() && ir.hasLeftChildren()) || (ir.parentBB->isPipe() && annotated[reinterpret_cast<ff_pipeline*>(ir.parentBB)->get_laststage()] != gName));
      });
    return groupNames;
  }

  std::set<std::string> inputGroups(std::set<std::string> groupNames){
    if (groupNames.size() > 1) 
      std::erase_if(groupNames,[this](const auto& gName){
        ff_IR& ir = annotatedGroups[gName];
        return ((ir.parentBB->isAll2All() && ir.isVertical() && annotatedGroups[gName].hasRightChildren()) || (ir.parentBB->isPipe() && annotated[reinterpret_cast<ff_pipeline*>(ir.parentBB)->get_firststage()] != gName));
      });
    return groupNames;
  }

};



static inline int DFF_Init(int& argc, char**& argv){
    struct sigaction s;
    memset(&s,0,sizeof(s));    
    s.sa_handler=SIG_IGN;
    if ( (sigaction(SIGPIPE,&s,NULL) ) == -1 ) {   
		perror("sigaction");
		return -1;
    } 


	std::string configFile, groupName;  

    for(int i = 0; i < argc; i++){
      if (strstr(argv[i], "--DFF_Config") != NULL){
        char * equalPosition = strchr(argv[i], '=');
        if (equalPosition == NULL){
          // the option is in the next argument array position
          configFile = std::string(argv[i+1]);
          argv[i] = argv[i+1] = NULL;
          i++;
        } else {
          // the option is in the next position of this string
          configFile = std::string(++equalPosition);
          argv[i] = NULL;
        }
        continue;
      }


      if (strstr(argv[i], "--DFF_GName") != NULL){
        char * equalPosition = strchr(argv[i], '=');
        if (equalPosition == NULL){
          // the option is in the next argument array position
          groupName = std::string(argv[i+1]);
          argv[i] = argv[i+1] = NULL;
          i++;
        } else {
          // the option is in the next position of this string
          groupName = std::string(++equalPosition);
          argv[i] = NULL;
        }
        continue;
      } 
    }

    if (configFile.empty()){
      ff::error("Config file not passed as argument!\nUse option --DFF_Config=\"config-file-name\"\n");
      return -1;
    }
    
    dGroups::Instance()->parseConfig(configFile);

    if (!groupName.empty())
      dGroups::Instance()->forceProtocol(Proto::TCP);
  #ifdef DFF_MPI
    else
        dGroups::Instance()->forceProtocol(Proto::MPI);
  #endif


    if (dGroups::Instance()->usedProtocol == Proto::TCP){
       if (groupName.empty()){
        ff::error("Group not passed as argument!\nUse option --DFF_GName=\"group-name\"\n");
        return -1;
      } 
      dGroups::Instance()->setRunningGroup(groupName); 
    }

  #ifdef DFF_MPI
    if (dGroups::Instance()->usedProtocol == Proto::MPI){
      //MPI_Init(&argc, &argv);
      int provided;
      
      if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS)
        return -1;
      
      
      // no thread support 
      if (provided < MPI_THREAD_MULTIPLE){
          error("No thread support by MPI\n");
          return -1;
      }

      int myrank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      dGroups::Instance()->setRunningGroupByRank(myrank);

      std::cout << "Running group: " << dGroups::Instance()->getRunningGroup() << " on rank: " <<  myrank << "\n";
    }

  #endif 

    // set the mapping if specified
    dGroups::Instance()->setThreadMapping();

    // set the name for the printer
    ff::cout.setPrefix(dGroups::Instance()->getRunningGroup());

    // recompact the argv array
    int j = 0;
    for(int i = 0;  i < argc; i++)
      if (argv[i] != NULL)
        argv[j++] = argv[i];
    
    // update the argc value
    argc = j;

    return 0;
}

static inline const std::string DFF_getMyGroup() {
	return dGroups::Instance()->getRunningGroup();
}

int ff_pipeline::run_and_wait_end() {
    dGroups::Instance()->run_and_wait_end(this);
    return 0;
}

int ff_a2a::run_and_wait_end(){
  ff_pipeline p;
  p.add_stage(this);
  dGroups::Instance()->run_and_wait_end(&p);
  return 0;
}



	
}
#endif
