/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#include <paddle/internals/dserver/DataClient.h>
#include <paddle/internals/dserver/DataServer.h>
#include <gtest/gtest.h>
#include <paddle/utils/Flags.h>
#include <paddle/utils/Util.h>

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

P_DECLARE_int32(num_gradient_servers);

class DataServerTester : public DataServer {
public:
  explicit DataServerTester(int port) : DataServer(string(), port) {}
  ~DataServerTester() {}
  void setup() { CHECK(DataServer::init()); }

  template <class T>
  void sendDataTest(SendDataType type, size_t size, DataUpdateMode mode1,
                    DataUpdateMode mode2);

  void sendGradTest(SendDataType type, size_t size, DataUpdateMode mode1,
                    DataUpdateMode mode2);
  void synchronizeTest();
};

template <class T>
void DataServerTester::sendDataTest(SendDataType type, size_t size,
                                    DataUpdateMode mode1,
                                    DataUpdateMode mode2) {
  DataClient client1, client2, client3;
  client1.init();
  client2.init();
  client3.init();

  ThreadWorker worker1;
  ThreadWorker worker2;
  ThreadWorker worker3;

  T* testData1 = new T[size];
  T* testData2 = new T[size];
  T* testData3 = new T[size];

  T* getData = new T[size * 3];
  T* getDataExpect = new T[size];

  /*data init and compute dest data*/
  // TODO(yuyang18): Change rand to rand_r for thread safety.
  for (size_t i = 0; i < size; ++i) {
    if (typeid(T) == typeid(real)) {
      testData1[i] = 1.0 * rand() / RAND_MAX;  // NOLINT
      testData2[i] = 1.0 * rand() / RAND_MAX;  // NOLINT
      testData3[i] = 1.0 * rand() / RAND_MAX;  // NOLINT
    } else {
      testData1[i] = rand();  // NOLINT
      testData2[i] = rand();  // NOLINT
      testData3[i] = rand();  // NOLINT
    }
    getDataExpect[i] = testData1[i];
  }

  auto put1 = [&]() {
    LOG(INFO) << "putData1 start";
    client1.putData<T>(0, type, testData1, size, mode1);
    LOG(INFO) << "putData1 finish";
  };

  auto put2 = [&]() {
    LOG(INFO) << "putData2 start";
    client2.putData<T>(1, type, testData2, size, mode1);
    LOG(INFO) << "putData2 finish";
  };

  auto put3 = [&]() {
    LOG(INFO) << "putData3 start";
    client3.putData<T>(2, type, testData3, size, mode1);
    LOG(INFO) << "putData3 finish";
  };

  auto get1 = [&]() {
    LOG(INFO) << "get ref start";
    client1.getData<T>(0, type, getData, size * 3, mode2);
    for (size_t i = 0; i < size; ++i) {
      CHECK_EQ(getData[i], getDataExpect[i]);
      CHECK_EQ(getData[i + size], testData2[i]);
      CHECK_EQ(getData[i + 2 * size], testData3[i]);
    }
    LOG(INFO) << "get ref finish";
  };
  worker1.addJob(put1);
  worker1.addJob(get1);
  worker2.addJob(put2);
  worker3.addJob(put3);

  worker1.addJob(put1);
  worker2.addJob(put2);
  worker3.addJob(put3);
  worker1.addJob(get1);

  worker1.wait();
  worker2.wait();
  worker3.wait();
  free(testData1);
  free(testData2);
  free(testData3);
  free(getDataExpect);
  free(getData);
}

void DataServerTester::sendGradTest(SendDataType type, size_t size,
                                    DataUpdateMode mode1,
                                    DataUpdateMode mode2) {
  DataClient client1, client2, client3;
  client1.init();
  client2.init();
  client3.init();

  ThreadWorker worker1;
  ThreadWorker worker2;
  ThreadWorker worker3;

  real* testData1 = new real[size];
  real* testData2 = new real[size];
  real* testData3 = new real[size];

  real* getData = new real[size / 3];
  real* getDataExpect = new real[size / 3];

  /*data init and compute dest data*/
  // TODO(yuyang18): Change rand to rand_r for thread safety
  for (size_t i = 0; i < size; ++i) {
    testData1[i] = 1.0 * rand() / RAND_MAX;  // NOLINT
    testData2[i] = 1.0 * rand() / RAND_MAX;  // NOLINT
    testData3[i] = 1.0 * rand() / RAND_MAX;  // NOLINT
  }

  for (size_t i = 0; i < size / 3; ++i) {
    getDataExpect[i] = testData1[i] + testData2[i] + testData3[i];
  }

  auto put1 = [&]() {
    LOG(INFO) << "putData1 start";
    client1.putData<real>(0, type, testData1, size, mode1);
    LOG(INFO) << "putData1 finish";
  };

  auto put2 = [&]() {
    LOG(INFO) << "putData2 start";
    client2.putData<real>(1, type, testData2, size, mode1);
    LOG(INFO) << "putData2 finish";
  };

  auto put3 = [&]() {
    LOG(INFO) << "putData3 start";
    client3.putData<real>(2, type, testData3, size, mode1);
    LOG(INFO) << "putData3 finish";
  };

  auto get1 = [&]() {
    LOG(INFO) << "get ref grad start";
    client1.getData<real>(0, type, getData, size / 3, mode2);
    for (size_t i = 0; i < size / 3; ++i) {
      CHECK_EQ(getData[i], getDataExpect[i]);
    }
    LOG(INFO) << "get ref grad finish";
  };
  worker1.addJob(put1);
  worker1.addJob(get1);
  worker2.addJob(put2);
  worker3.addJob(put3);

  worker1.addJob(put1);
  worker2.addJob(put2);
  worker3.addJob(put3);
  worker1.addJob(get1);

  worker1.wait();
  worker2.wait();
  worker3.wait();
  free(testData1);
  free(testData2);
  free(testData3);
  free(getDataExpect);
  free(getData);
}

TEST(DataServer, sendData) {
  // Set gserver and pserver all 3, so that the test is sufficient.
  int oldFlagsPortsNum = FLAGS_ports_num;
  int oldFlagsNumGradientServers = FLAGS_num_gradient_servers;
  FLAGS_ports_num = 3;
  FLAGS_num_gradient_servers = 3;
  std::unique_ptr<DataServerTester> g_server1;
  std::unique_ptr<DataServerTester> g_server2;
  std::unique_ptr<DataServerTester> g_server3;
  g_server1.reset(new DataServerTester(FLAGS_data_server_port));
  g_server1->start();
  g_server2.reset(new DataServerTester(FLAGS_data_server_port + 1));
  g_server2->start();
  g_server3.reset(new DataServerTester(FLAGS_data_server_port + 2));
  g_server3->start();

  g_server2->init();
  g_server3->init();

  g_server1->setup();
  sleep(1);
  g_server1->sendDataTest<real>(DATA_REF, 256, DATA_UPDATE_MODE_SET_REF,
                                DATA_UPDATE_MODE_GET_REF);
  g_server1->sendDataTest<int>(DATA_REFLABEL, 256,
                               DATA_UPDATE_MODE_SET_REF_LABEL,
                               DATA_UPDATE_MODE_GET_REF_LABEL);
  g_server1->sendGradTest(DATA_REFGRAD, 768, DATA_UPDATE_MODE_SET_REF_GRAD,
                          DATA_UPDATE_MODE_GET_REF_GRAD);

  g_server1.reset();
  g_server2.reset();
  g_server3.reset();

  FLAGS_ports_num = oldFlagsPortsNum;
  FLAGS_num_gradient_servers = oldFlagsNumGradientServers;
}

int main(int argc, char** argv) {
  paddle::initMain(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
