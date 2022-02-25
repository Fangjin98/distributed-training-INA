import logging
import time
import os
import json

from bfruntime_client_base_tests import BfRuntimeTest
import bfrt_grpc.client as gc

logger = logging.getLogger('Test')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RULE_PATH = os.path.join(BASE_DIR, 'rules.json')
if not os.path.exists(RULE_PATH):
    logger.error('Please link one of existing rules*.json as rules.json')
    sys.exit(-1)


class MyBfRuntimeTest(BfRuntimeTest):
    def entry_add(self, table, key_list, data_list):
        try:
            #table.entry_del(self.target, key_list)
            print('Remove non-existing entry')
        except:
            pass
        print('entry_add:  {key: %s, data: %s}' % (key_list, data_list))
        table.entry_add(self.target, key_list, data_list)


class ruleTest(MyBfRuntimeTest):
    def setUp(self):
        self.p4_name = "nga"
        self.rules = json.loads(open(RULE_PATH).read())
        self.target = gc.Target(device_id=0, pipe_id=0xffff)
        BfRuntimeTest.setUp(self, client_id=0, p4_name=self.p4_name)

    def tearDown(self):
        BfRuntimeTest.tearDown(self)

    def runTest(self):
        logger.info('Adding rule')

        bfrt_info = self.interface.bfrt_info_get(self.p4_name)

        self.config_forward(bfrt_info)

    def config_forward(self, bfrt_info):
        table_name = 'IngressPipeline.forward'
        table = bfrt_info.table_get(table_name)
        table.info.key_field_annotation_add("hdr.ipv4.dst_addr", "ipv4")
        table.info.data_field_annotation_add("dst_addr", "ipv4_forward", "mac")
        for rule in self.rules[table_name]:
            key = table.make_key(
                [
                    gc.KeyTuple('hdr.ipv4.dst_addr',
                                rule['key']['dst_addr'])
                ]
            )
            data = table.make_data(
                [
                    gc.DataTuple('dst_addr', rule['data']['dst_addr']),
                    gc.DataTuple('dst_port', rule['data']['dst_port']),
                ],
                'IngressPipeline.ipv4_forward'
            )
            self.entry_add(table, [key], [data])