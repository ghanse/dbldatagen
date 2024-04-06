from .dataset_provider import DatasetProvider, dataset_definition


@dataset_definition(name="json/device-status",
                    summary="JSON Device Status Data Set",
                    autoRegister=True)
class JSONDeviceStatusProvider(DatasetProvider):
    """ Basic User Data Set

    This is a basic user data set with customer id, name, email, ip address, and phone number.

    """
    MAX_LONG = 9223372036854775807
    ALLOWED_OPTIONS = ["random", "dummyValues", "rows", "partitions", "tableName"]

    def getTable(self, sparkSession, *, tableName=None, rows=-1, partitions=-1,
                 **options):
        import dbldatagen as dg

        generateRandom = options.get("random", False)
        dummyValues = options.get("dummyValues", 0)

        if rows is None or rows < 0:
            rows = 100000

        if partitions is None or partitions < 0:
            partitions = self.autoComputePartitions(rows, 8)

        assert tableName is None or tableName == "primary", "Invalid table name"
        df_spec = (
            dg.DataGenerator(sparkSession=sparkSession, name="test_data_set1", rows=rows,
                             partitions=partitions, randomSeedMethod="hash_fieldname")
            .withColumn("customer_id", "long", minValue=1000000, random=generateRandom)
            .withColumn("name", "string",
                        template=r'\w \w|\w \w \w', random=generateRandom)
            .withColumn("email", "string",
                        template=r'\w.\w@\w.com|\w@\w.co.u\k', random=generateRandom)
            .withColumn("ip_addr", "string",
                        template=r'\n.\n.\n.\n', random=generateRandom)
            .withColumn("phone", "string",
                        template=r'(ddd)-ddd-dddd|1(ddd) ddd-dddd|ddd ddddddd',
                        random=generateRandom)
        )

        if dummyValues > 0:
            df_spec = df_spec.withColumn("dummy", "long", random=True, numColumns=dummyValues, minValue=1)

        return df_spec
