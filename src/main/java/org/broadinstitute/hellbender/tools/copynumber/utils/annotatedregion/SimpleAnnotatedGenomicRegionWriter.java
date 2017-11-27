package org.broadinstitute.hellbender.tools.copynumber.utils.annotatedregion;

import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.tsv.DataLine;
import org.broadinstitute.hellbender.utils.tsv.TableColumnCollection;
import org.broadinstitute.hellbender.utils.tsv.TableWriter;
import org.spark_project.guava.collect.Lists;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;

public class SimpleAnnotatedGenomicRegionWriter implements AnnotatedGenomicRegionWriter  {

    private SimpleTableWriter writer;
    private String contigColumnHeader;
    private String startColumnHeader;
    private String endColumnHeader;

    private class SimpleTableWriter extends TableWriter<SimpleAnnotatedGenomicRegion> {

        public SimpleTableWriter(File file, TableColumnCollection tableColumns) throws IOException {
            super(file, tableColumns);
        }


        @Override
        protected void composeLine(SimpleAnnotatedGenomicRegion record, DataLine dataLine) {
            // First the Locatable info
            dataLine.set(contigColumnHeader, record.getContig());
            dataLine.set(startColumnHeader, record.getStart());
            dataLine.set(endColumnHeader, record.getEnd());

            // Now everything else.
            record.getAnnotations().keySet().stream().forEach(k -> dataLine.set(k, record.getAnnotationValue(k)));
        }
    }

    /**
     * Initialize a writer for the given collection to the given output file.
     *
     * @param collection collection to write.  Can't be {@code null}
     * @param outputFile destination file.  Must be writeable.
     */
    public SimpleAnnotatedGenomicRegionWriter(final SimpleAnnotatedGenomicRegionCollection collection, final File outputFile) {
        Utils.nonNull(collection);
        Files.isWritable(outputFile.toPath());

        final List<String> finalColumnList = Lists.newArrayList(collection.getContigColumnName(), collection.getStartColumnName(), collection.getEndColumnName());
        finalColumnList.addAll(collection.getAnnotations());
        try {
            writer = new SimpleTableWriter(outputFile, new TableColumnCollection(finalColumnList));
        } catch (final IOException ioe) {
            throw new UserException.CouldNotCreateOutputFile(outputFile, "Could not create: " + outputFile.getAbsolutePath());
        }

        this.contigColumnHeader = collection.getContigColumnName();
        this.startColumnHeader = collection.getStartColumnName();
        this.endColumnHeader = collection.getEndColumnName();
    }

    @Override
    public void writeHeader(final String SAMFileHeader, final List<String> comments, final List<String> annotations,
                            final String contigColumnName, final String startColumnName, final String endColumnName) {
        try {
            writer.writeHeaderIfApplies();
        } catch (final IOException e) {
            throw new UserException.CouldNotCreateOutputFile("Could not write to file.", e);
        }
    }

    @Override
    public void close() {
        try {
            writer.close();
        } catch (final IOException e) {
            throw new UserException.CouldNotCreateOutputFile("Could not close file writing.", e);
        }
    }

    @Override
    public void add(SimpleAnnotatedGenomicRegion simpleAnnotatedGenomicRegion) {
        try {
            writer.writeRecord(simpleAnnotatedGenomicRegion);
        } catch (IOException e) {
            throw new UserException.CouldNotCreateOutputFile("Could not write to file.", e);
        }
    }
}
