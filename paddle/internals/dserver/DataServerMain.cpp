/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#include <fstream>
#include "paddle/utils/Flags.h"
#include "paddle/utils/StringUtil.h"
#include "paddle/utils/Util.h"
#include "DataServer.h"
#include "paddle/pserver/RDMANetwork.h"

using namespace paddle;  // NOLINT

int main(int argc, char** argv) {
  initMain(argc, argv);

  std::vector<std::string> devices;
  std::vector<std::shared_ptr<DataServer>> dservers;

 // round robin to loadbalance RDMA server ENGINE
 int rdmaCpu = 0;
 int onlineCpus = rdma::numCpus();

  if (FLAGS_nics.empty()) {
    dservers.resize(FLAGS_ports_num);
    for (int i = 0; i < FLAGS_ports_num; ++i) {
      if (FLAGS_rdma_tcp == "rdma") {
         dservers[i].reset(
             new DataServer(std::string(), FLAGS_data_server_port + i,
               rdmaCpu++));
         rdmaCpu = rdmaCpu % onlineCpus;
       } else {
         dservers[i].reset(
           new DataServer(std::string(), FLAGS_data_server_port + i));
       }
      CHECK(dservers[i]->init()) << "Fail to initialize parameter server"
                    << FLAGS_data_server_port + FLAGS_ports_num + i;
      LOG(INFO) << "dserver started : " << FLAGS_data_server_port + i;
      dservers[i]->start();
    }
  } else {
    str::split(FLAGS_nics, ',', &devices);
    dservers.resize(devices.size() * FLAGS_ports_num);
    for (int i = 0; i < FLAGS_ports_num; ++i) {
      for (size_t j = 0; j < devices.size(); ++j) {
        if (FLAGS_rdma_tcp == "rdma") {
           dservers[i * devices.size() + j].reset(
               new DataServer(getIpAddr(devices[j]), FLAGS_data_server_port + i,
                 rdmaCpu++));
           rdmaCpu = rdmaCpu % onlineCpus;
        } else {
           dservers[i * devices.size() + j].reset(
             new DataServer(getIpAddr(devices[j]), FLAGS_data_server_port + i));
        }
        CHECK(dservers[i * devices.size() + j]->init())
            << "Fail to initialize data server" << devices[j]
            << FLAGS_data_server_port + i;
        LOG(INFO) << "dserver started : " << devices[j] << ":"
                  << FLAGS_data_server_port + i;
        dservers[i * devices.size() + j]->start();
      }
    }
  }

  for (auto& dserver : dservers) {
    dserver->join();
  }

  return 0;
}
