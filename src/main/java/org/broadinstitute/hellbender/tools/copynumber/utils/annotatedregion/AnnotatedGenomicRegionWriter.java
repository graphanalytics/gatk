package org.broadinstitute.hellbender.tools.copynumber.utils.annotatedregion;

import java.io.Closeable;
import java.util.List;

public interface AnnotatedGenomicRegionWriter extends Closeable {

    /**
     * Write only the header (and any SAMFileHeader or comments)
     *
     * @param SAMFileHeader SAM file header as a multiline string.  {@code null} is allowed, if not available.
     * @param comments Comments to prepend to the xsv file.  Use an empty list, if no comments are needed.
     * @param annotations annotation names that do not include the locatable column names.
     * @param contigColumnName how contig should be rendered
     * @param startColumnName how start position should be rendered
     * @param endColumnName how end position should be rendered
     */
    void writeHeader(final String SAMFileHeader, final List<String> comments, final List<String> annotations,
                     final String contigColumnName, final String startColumnName, final String endColumnName);

    /**
     * attempt to close the file
     */
    @Override
    void close() ;

    /** Write one region to the file.
     *
     * @param simpleAnnotatedGenomicRegion region to write
     */
    void add(final SimpleAnnotatedGenomicRegion simpleAnnotatedGenomicRegion);
}
