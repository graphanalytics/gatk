package org.broadinstitute.hellbender.tools.copynumber.utils.annotatedregion;

import com.google.common.collect.Sets;
import htsjdk.samtools.SAMFileHeader;
import htsjdk.tribble.readers.AsciiLineReader;
import htsjdk.tribble.readers.AsciiLineReaderIterator;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.codecs.xsvLocatableTable.XsvLocatableTableCodec;
import org.broadinstitute.hellbender.utils.codecs.xsvLocatableTable.XsvTableFeature;
import org.broadinstitute.hellbender.utils.io.IOUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Represents a collection of annotated regions.  The annotations do not need to be known ahead of time, if reading from a file.
 * Though {@link SimpleAnnotatedGenomicRegion::CONTIG_HEADER},
 *  {@link SimpleAnnotatedGenomicRegion::START_HEADER}, and
 *  {@link SimpleAnnotatedGenomicRegion::END_HEADER} are expected, by default, for defining the genomic region.
 *
 *  This class supports reading xsv (tsv, csv) files with comments ("#") and SAM headers ("@").  The default is tsv.
 */
public class SimpleAnnotatedGenomicRegionCollection {

    static final Logger logger = LogManager.getLogger(SimpleAnnotatedGenomicRegionCollection.class);

    public static final String ANNOTATED_REGION_DEFAULT_CONFIG_RESOURCE = "org/broadinstitute/hellbender/tools/copynumber/utils/annotatedregion/annotated_region_default.config";
    private SAMFileHeader samFileHeader;

    /** Does not include the locatable fields. */
    private List<String> annotations;
    private List<String> comments;
    private List<SimpleAnnotatedGenomicRegion> records;

    /* The output contig column name */
    private String contigColumnName;

    /* The output start column name */
    private String startColumnName;

    /* The output end column name */
    private String endColumnName;

    private SimpleAnnotatedGenomicRegionCollection(final SAMFileHeader samFileHeader, final List<String> annotations, final List<String> comments,
                                                   final List<SimpleAnnotatedGenomicRegion> records, final String contigColumnName,
                                                   final String startColumnName, final String endColumnName) {
        this.comments = comments;
        this.samFileHeader = samFileHeader;
        this.annotations = annotations;
        this.records = records;
        this.contigColumnName = contigColumnName;
        this.startColumnName = startColumnName;
        this.endColumnName = endColumnName;
    }

    /**
     *  Same as {@link #create(Path, Path, Set)} , but uses the default annotation
     *   region config file in the GATK.
     * @param input See {@link #create(Path, Path, Set)}
     * @param headersOfInterest See {@link #create(Path, Path, Set)}
     * @return See {@link #create(Path, Path, Set)}
     */
    public static SimpleAnnotatedGenomicRegionCollection create(final Path input, final Set<String> headersOfInterest) {
        ClassLoader classLoader = SimpleAnnotatedGenomicRegionCollection.class.getClassLoader();
        final File defaultConfigFile = new File(classLoader.getResource(ANNOTATED_REGION_DEFAULT_CONFIG_RESOURCE).getFile());
        return create(input, defaultConfigFile.toPath(), headersOfInterest);
    }

    /**
     * Same as {@link #create(Path, Path, Set)} , but uses the default annotation
     *   region config file in the GATK.
     *
     *   This is just a convenience method to use a File instead of Path.
     *
     * @param input See {@link #create(Path, Path, Set)}
     * @param headersOfInterest See {@link #create(Path, Path, Set)}
     * @return See {@link #create(Path, Path, Set)}
     */
    public static SimpleAnnotatedGenomicRegionCollection create(final File input, final Set<String> headersOfInterest) {
        return create(input.toPath(), headersOfInterest);
    }

    /**
     *  Create a new collection from the metadata of an existing collection and a list of {@link SimpleAnnotatedGenomicRegion}
     *
     * @param regions new regions to use in the resulting collection.  Never {@code null}.
     * @param collection existing collection to use for all attributes, except the list of records.  Never {@code null}.
     * @return a collection that is identical to the input collection, except that it uses the input regions for records.
     */
    public static SimpleAnnotatedGenomicRegionCollection create(final List<SimpleAnnotatedGenomicRegion> regions,
                                                                final SimpleAnnotatedGenomicRegionCollection collection) {
        Utils.nonNull(regions);
        Utils.nonNull(collection);

        return new SimpleAnnotatedGenomicRegionCollection(collection.getSamFileHeader(), collection.getAnnotations(), Collections.emptyList(), regions,
                collection.getContigColumnName(), collection.getStartColumnName(), collection.getEndColumnName());
    }

    /**
     * Create a collection from components.
     *
     * @param regions regions to use in the resulting collection.  Never {@code null}.
     * @param samFileHeader SAMFileHeader to include in the collection.  Represents the sample(s)/references that were used for these segments.
     *                      {@code null} is allowed.
     * @param annotations List of annotations to preserve in the regions.  Never {@code null}.  These are the annotations that will be written.
     * @param contigColumnName contig column name to use if the collection is written.  Can't be {@code null}, empty, or a number.
     * @param startColumnName start column name to use if the collection is written.  Can't be {@code null}, empty, or a number.
     * @param endColumnName end column name to use if the collection is written.  Can't be {@code null}, empty, or a number.
     * @return collection based on the inputs
     */
    public static SimpleAnnotatedGenomicRegionCollection create(final List<SimpleAnnotatedGenomicRegion> regions,
                                                                                    final SAMFileHeader samFileHeader,
                                                                                    final List<String> annotations,
                                                                                    final String contigColumnName,
                                                                                    final String startColumnName,
                                                                                    final String endColumnName) {

        Utils.nonNull(regions);
        Utils.nonNull(annotations);
        XsvLocatableTableCodec.validateLocatableColumnName(contigColumnName);
        XsvLocatableTableCodec.validateLocatableColumnName(startColumnName);
        XsvLocatableTableCodec.validateLocatableColumnName(endColumnName);

        return new SimpleAnnotatedGenomicRegionCollection(samFileHeader, annotations, Collections.emptyList(), regions,
                contigColumnName, startColumnName, endColumnName);
    }



    /** Create a collection based on the contents of an input file and a given config file.  The config file must be the same as
     * is ingested by {@link XsvLocatableTableCodec}.
     *
     * @param input readable path to use for the xsv file.  Must be readable.  Never {@code null}.
     * @param inputConfigFile config file for specifying the format of the xsv file.  Must be readable.  Never {@code null}.
     * @param headersOfInterest Only preserve these headers.  These must be present in the input file.  This parameter should not include the locatable columns
     *                          defined by the config file, which are always preserved.
     *                          Use {@code null} to indicate "all headers are of interest".
     * @return never {@code null}
     */
    public static SimpleAnnotatedGenomicRegionCollection create(final Path input, final Path inputConfigFile, final Set<String> headersOfInterest) {

        IOUtils.assertFileIsReadable(input);
        IOUtils.assertFileIsReadable(inputConfigFile);

        final XsvLocatableTableCodec codec = new XsvLocatableTableCodec(inputConfigFile);
        final List<SimpleAnnotatedGenomicRegion> regions = new ArrayList<>();

        if (codec.canDecode(input.toString())) {
            try (final InputStream fileInputStream = Files.newInputStream(input)) {

                // Lots of scaffolding to do reading here:
                final AsciiLineReaderIterator lineReaderIterator = new AsciiLineReaderIterator(AsciiLineReader.from(fileInputStream));
                final List<String> header = codec.readActualHeader(lineReaderIterator);
                warnAllHeadersOfInterestNotPresent(headersOfInterest, header);

                final List<String> featureCols = codec.getHeaderWithoutLocationColumns();

                while (lineReaderIterator.hasNext()) {
                    final XsvTableFeature feature = codec.decode(lineReaderIterator.next());
                    if (feature == null) {
                        continue;
                    }

                    final List<String> featureValues = feature.getValuesWithoutLocationColumns();

                    final SortedMap<String, String> annotations = new TreeMap<>();
                    IntStream.range(0, featureCols.size()).boxed()
                            .filter(i -> (headersOfInterest == null) || headersOfInterest.contains(featureCols.get(i)))
                            .forEach(i -> annotations.put(featureCols.get(i), featureValues.get(i)));

                    regions.add(new SimpleAnnotatedGenomicRegion(
                            new SimpleInterval(feature.getContig(), feature.getStart(), feature.getEnd()),
                            annotations));
                }

                return new SimpleAnnotatedGenomicRegionCollection(codec.createSamFileHeader(), codec.getHeaderWithoutLocationColumns(),
                        codec.getComments(), regions, codec.getFinalContigColumn(), codec.getFinalStartColumn(), codec.getFinalEndColumn());

            }
            catch ( final FileNotFoundException ex ) {
                throw new GATKException("Error - could not find test file: " + input, ex);
            }
            catch ( final IOException ex ) {
                throw new GATKException("Error - IO problem with file " + input, ex);
            }
        }
        else {
            throw new UserException.BadInput("Could not parse xsv file.");
        }
    }

    private static void warnAllHeadersOfInterestNotPresent(final Set<String> headersOfInterest, final List<String> header) {
        if ((headersOfInterest != null) && !header.containsAll(headersOfInterest)) {
            final Set<String> unusedColumnsOfInterest = Sets.difference(new HashSet<>(headersOfInterest), new HashSet<>(header));
            if (unusedColumnsOfInterest.size() > 0) {
                final List<String> missingColumns = new ArrayList<>(unusedColumnsOfInterest);
                logger.warn("Some columns of interest specified by the user were not seen in any input files: " + StringUtils.join(missingColumns, ", "));
            }
        }
    }

    /**
     *  Write this collection to a file
     * @param outputFile destination file, must be writable.
     */
    public void write(final File outputFile) {

        Utils.validateArg(Files.isWritable(outputFile.toPath()), "Can not write to: " + outputFile.getAbsolutePath());

        final AnnotatedGenomicRegionWriter writer = new SimpleAnnotatedGenomicRegionWriter(this, outputFile);
        writer.writeHeader(getSamFileHeader().getSAMString(), getComments(), getAnnotations(), contigColumnName,
                startColumnName, endColumnName);
        getRecords().forEach(writer::add);
        writer.close();
    }

    /** Can return {@code null} */
    public SAMFileHeader getSamFileHeader() {
        return samFileHeader;
    }

    public List<String> getAnnotations() {
        return annotations;
    }

    public List<String> getComments() {
        return comments;
    }

    public List<SimpleAnnotatedGenomicRegion> getRecords() {
        return records;
    }

    public int size() {
        return getRecords().size();
    }

    public String getContigColumnName() {
        return contigColumnName;
    }

    public String getStartColumnName() {
        return startColumnName;
    }

    public String getEndColumnName() {
        return endColumnName;
    }
}
