# Generic substitution

Generic integer and decimal surrogates preserve Spark types and equality relationships, but they do not preserve the
original values' magnitudes. Generic floating-point substitution rejects NaN and positive or negative infinity with a
column-specific error; only finite values are supported.
