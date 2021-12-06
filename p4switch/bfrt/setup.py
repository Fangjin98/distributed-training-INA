from ipaddress import ip_address

p4 = bfrt.nga.pipe

ip01="172.16.170.1"
ip02="172.16.170.2"
ip03="172.16.170.3"

mac01=0x48df37aafaa8
mac02=0x48df375cffb8
mac03=0x48df37aaacb8

port01=132
port02=133
port03=134

def ip2int(ip):
    ip_list = ip.strip().split('.')
    ip_int = int(ip_list[0])*256**3+int(ip_list[1])*256**2+int(ip_list[2])*256**1+int(ip_list[3])*256**0
    return ip_int

# This function can clear all the tables and later on other fixed objects
# once bfrt support is added.
def clear_all(verbose=True, batching=True):
    global p4
    global bfrt
    
    def _clear(table, verbose=False, batching=False):
        if verbose:
            print("Clearing table {:<40} ... ".
                  format(table['full_name']), end='', flush=True)
        try:    
            entries = table['node'].get(regex=True, print_ents=False)
            try:
                if batching:
                    bfrt.batch_begin()
                for entry in entries:
                    entry.remove()
            except Exception as e:
                print("Problem clearing table {}: {}".format(
                    table['name'], e.sts))
            finally:
                if batching:
                    bfrt.batch_end()
        except Exception as e:
            if e.sts == 6:
                if verbose:
                    print('(Empty) ', end='')
        finally:
            if verbose:
                print('Done')

        # Optionally reset the default action, but not all tables
        # have that
        try:
            table['node'].reset_default()
        except:
            pass
    
    # The order is important. We do want to clear from the top, i.e.
    # delete objects that use other objects, e.g. table entries use
    # selector groups and selector groups use action profile members
    

    # Clear Match Tables
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['MATCH_DIRECT', 'MATCH_INDIRECT_SELECTOR']:
            _clear(table, verbose=verbose, batching=batching)

    # Clear Selectors
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['SELECTOR']:
            _clear(table, verbose=verbose, batching=batching)
            
    # Clear Action Profiles
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['ACTION_PROFILE']:
            _clear(table, verbose=verbose, batching=batching)
    
clear_all()

switch_check = p4.Ingress.switch_check
ipv4_lpm = p4.Ingress.ipv4_lpm

switch_check.add_with_set_agg(b'00000000')
print("done")
ipv4_lpm.add_with_ipv4_forward(dstAddr=ip_address(ip01),dstMacAddr=mac01,port=port01)  # dstip, dstmac, port
print("done")
ipv4_lpm.add_with_ipv4_forward(dstAddr=ip_address(ip02),dstMacAddr=mac02,port=port02)
print("done")
ipv4_lpm.add_with_ipv4_forward(dstAddr=ip_address(ip03),dstMacAddr=mac03,port=port03)
print("done")

# register_table_size = p4.Ingress.table_size_reg
# register_counter = p4.Ingress.test_reg

# set the size of the table
# register_table_size.mod(register_index=0,f1=table_size) 
# start from the first server
# register_counter.mod(register_index=0,f1=table_size-1)

# clean the counters
def clear_counters(table_node):
    for e in table_node.get(regex=True):
        e.data[b'$COUNTER_SPEC_BYTES'] = 0
        e.data[b'$COUNTER_SPEC_PKTS'] = 0
        e.push()

# dump everything
switch_check.dump(table=True)
ipv4_lpm.dump(table=True)
# register_table_size.dump(table=True,from_hw=1)
# register_counter.dump(table=True,from_hw=1)