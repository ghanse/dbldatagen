import dbldatagen as dg
from . import DatasetProvider, dataset_definition


@dataset_definition(name="json/device-status",
                    summary="JSON Device Status Data Set",
                    autoRegister=True)
class JSONDeviceStatusProvider(DatasetProvider):
    """ Basic User Data Set

    This is a basic user data set with customer id, name, email, ip address, and phone number.

    """

    def getTable(self, rows=1000000, partitions=-1, dummyValues=0, random=False):
        df_spec = (
             dg.DataGenerator(sparkSession=spark, name="test_data_set1", rows=100000,
                              partitions=4, randomSeedMethod="hash_fieldname")
            .withIdOutput()
            .withColumnSpec("email",
                            template=r'\w.\w@\w.com|\w@\w.co.u\k')
            .withColumnSpec("ip_addr",
                             template=r'\n.\n.\n.\n')
            .withColumnSpec("phone",
                             template=r'(ddd)-ddd-dddd|1(ddd) ddd-dddd|ddd ddddddd')
            )

        return df_spec