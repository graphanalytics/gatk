package org.broadinstitute.hellbender.tools.spark.sv.discovery.prototype;

import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.BetaFeature;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.hellbender.cmdline.CommandLineProgram;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.cmdline.programgroups.StructuralVariantDiscoveryProgramGroup;
import org.broadinstitute.hellbender.tools.spark.sv.utils.*;
import org.broadinstitute.hellbender.tools.spark.sv.utils.SVKmer.Base;
import org.broadinstitute.hellbender.tools.spark.sv.utils.SVFastqUtils.FastqRead;
import scala.Tuple2;

import java.util.*;

@BetaFeature
@CommandLineProgramProperties(
        oneLineSummary = "(Internal) junk", summary = "complete crap",
        programGroup = StructuralVariantDiscoveryProgramGroup.class)
public class KmerAdjacencyBuilder extends CommandLineProgram {
    private static final long serialVersionUID = 1L;

    @Argument(doc = "input fastq",
            shortName = StandardArgumentDefinitions.INPUT_SHORT_NAME,
            fullName = StandardArgumentDefinitions.INPUT_LONG_NAME)
    public String fastqFile;

    @Override protected Object doWork() {
        final List<FastqRead> reads = SVFastqUtils.readFastqFile(fastqFile);
        final int kSize = 63;
        final int minQ = 10;
        final int minKCount = 5;

        // kmerize each read, counting the observations of each kmer.
        // trim each read at the first base having a quality less than minQ
        final int nKmers = reads.stream().mapToInt(read -> Math.min(0, read.getBases().length - kSize + 1)).sum();
        final Map<SVKmerLong, Integer> kmerCounts = new HashMap<>(SVUtils.hashMapCapacity(nKmers));
        for ( final FastqRead read : reads ) {
            final byte[] quals = read.getQuals();
            int trimLen;
            for ( trimLen = 0; trimLen != quals.length; ++trimLen ) {
                if ( quals[trimLen] < minQ ) break;
            }
            SVKmerizer.canonicalStream(Arrays.copyOf(read.getBases(),trimLen),kSize,new SVKmerLong(kSize))
                    .forEach(kmer -> kmerCounts.merge((SVKmerLong)kmer, 1, Integer::sum));
        }

        // dump a histogram of kmer counts
        //final Map<Integer, Integer> kmerCountHistogram = new TreeMap<>();
        //kmerCounts.values().forEach( count -> kmerCountHistogram.merge(count, 1, Integer::sum));
        //kmerCountHistogram.forEach( (count, countCount) -> System.out.println(count + "\t" + countCount));

        // ignore kmers that appear less than minKCount times
        kmerCounts.entrySet().removeIf(entry -> entry.getValue() < minKCount);
        final Map<SVKmerLong, int[]> kmerAdjacencyMap = new HashMap<>(SVUtils.hashMapCapacity(kmerCounts.size()));
        kmerCounts.forEach((kmer, value) -> {
            final SVKmerLong kkk = kmer.removeFirstAndLastBase(kSize);
            final int[] counts = kmerAdjacencyMap.computeIfAbsent(kkk, key -> new int[8]);
            counts[kmer.firstBase(kSize).ordinal()] += value;
            counts[kmer.lastBase().ordinal() + 4] += value;
        });

        // the adjacency map won't include source and sink kmers
        // add them to the map now
        final int kSize2 = kSize - 2;
        final List<Tuple2<SVKmerLong, int[]>> sourcesAndSinks = new ArrayList<>();
        kmerAdjacencyMap.forEach((kmer, counts) -> {
            for ( final Base base : Base.values() ) {
                final int predCount = counts[base.ordinal()];
                if ( predCount > 0 ) {
                    final SVKmerLong predecessor = kmer.predecessor(base, kSize2);
                    final SVKmerLong canonicalPredecessor = predecessor.canonical(kSize2);
                    if ( !kmerAdjacencyMap.containsKey(canonicalPredecessor) ) {
                        final int[] newCounts = new int[8];
                        if ( predecessor.equals(canonicalPredecessor) ) { // if predecessor is in canonical form
                            newCounts[kmer.lastBase().ordinal() + 4] = predCount;
                        }
                        else {
                            newCounts[kmer.lastBase().complement().ordinal()] = predCount;
                        }
                        sourcesAndSinks.add(new Tuple2<>(canonicalPredecessor, newCounts));
                    }
                }
                final int succCount = counts[base.ordinal() + 4];
                if ( succCount > 0 ) {
                    final SVKmerLong successor = kmer.successor(base, kSize2);
                    final SVKmerLong canonicalSuccessor = successor.canonical(kSize2);
                    if ( !kmerAdjacencyMap.containsKey(canonicalSuccessor) ) {
                        final int[] newCounts = new int[8];
                        if ( successor.equals(canonicalSuccessor) ) { // if successor is in canonical form
                            newCounts[kmer.firstBase(kSize2).ordinal()] = succCount;
                        } else {
                            newCounts[4 + kmer.firstBase(kSize2).complement().ordinal()] = succCount;
                        }
                        sourcesAndSinks.add(new Tuple2<>(canonicalSuccessor, newCounts));
                    }
                }
            }
        });
        sourcesAndSinks.forEach(tuple2 ->
                kmerAdjacencyMap.merge(tuple2._1(), tuple2._2(), (oldCounts, newCounts) -> {
                    for ( int idx = 0; idx != oldCounts.length; ++idx ) {
                        oldCounts[idx] += newCounts[idx];
                    }
                    return oldCounts;
                }));

        // dump the adjacency map
        //for ( final Map.Entry<SVKmerLong, int[]> entry : kmerAdjacencyMap.entrySet() ) {
        //    final StringBuilder sb = new StringBuilder(entry.getKey().toString(kSize2));
        //    Arrays.stream(entry.getValue()).forEach(iii -> sb.append('\t').append(iii));
        //    System.out.println(sb);
        //}

        // build contigs
        final List<String> contigs = new ArrayList<>();
        kmerAdjacencyMap.forEach( (kmer, counts) -> {
            final SVKmerLong predecessor = getSolePredecessor(kmer, kSize2, counts);
            if ( predecessor == null || getSuccessorCount(predecessor, kSize2, kmerAdjacencyMap) > 1 ) {
                buildContig(kmer, kSize2, counts, kmerAdjacencyMap, contigs);
            } else {
                final SVKmerLong successor = getSoleSuccessor(kmer, kSize2, counts);
                if ( successor == null || getPredecessorCount(successor, kSize2, kmerAdjacencyMap) > 1 ) {
                    buildContig(kmer.reverseComplement(kSize2), kSize2, counts, kmerAdjacencyMap, contigs);
                }
            }
        });

        // dump contigs
        int contigId = 0;
        for ( final String contig : contigs ) {
            System.out.println(">contig"+contigId++);
            System.out.println(contig);
        }

        return null;
    }

    private static SVKmerLong getSolePredecessor( final SVKmerLong kmer, final int kSize, final int[] counts ) {
        Base solePred = null;
        final int offset = kmer.isCanonical(kSize) ? 0 : 4;
        for ( final Base base : Base.values() ) {
            if ( counts[base.ordinal() + offset] > 0 ) {
                if ( solePred != null ) return null; // found a 2nd predecessor
                solePred = base;
            }
        }
        if ( solePred == null ) return null;
        if ( !kmer.isCanonical(kSize) ) solePred = solePred.complement();
        return kmer.predecessor(solePred, kSize);
    }

    private static SVKmerLong getSoleSuccessor( final SVKmerLong kmer, final int kSize, final int[] counts ) {
        Base soleSucc = null;
        final int offset = kmer.isCanonical(kSize) ? 4 : 0;
        for ( final Base base : Base.values() ) {
            if ( counts[base.ordinal() + offset] > 0 ) {
                if ( soleSucc != null ) return null; // found a 2nd successor
                soleSucc = base;
            }
        }
        if ( soleSucc == null ) return null;
        if ( !kmer.isCanonical(kSize) ) soleSucc = soleSucc.complement();
        return kmer.successor(soleSucc, kSize);
    }

    private static int getPredecessorCount( final SVKmerLong kmer, final int kSize,
                                            final Map<SVKmerLong, int[]> kmerAdjacencyMap ) {
        if (!kmer.isCanonical(kSize)) {
            return getSuccessorCount(kmer.reverseComplement(kSize), kSize, kmerAdjacencyMap);
        }
        return getPredecessorCount(kmerAdjacencyMap.get(kmer));
    }

    private static int getPredecessorCount( final int[] counts ) {
        return (counts[0] > 0 ? 1 : 0) + (counts[1] > 0 ? 1 : 0) + (counts[2] > 0 ? 1 : 0) + (counts[3] > 0 ? 1 : 0);
    }

    private static int getSuccessorCount( final SVKmerLong kmer, final int kSize,
                                          final Map<SVKmerLong, int[]> kmerAdjacencyMap ) {
        if (!kmer.isCanonical(kSize)) {
            return getPredecessorCount(kmer.reverseComplement(kSize), kSize, kmerAdjacencyMap);
        }
        final int[] counts = kmerAdjacencyMap.get(kmer);
        return getSuccessorCount(kmerAdjacencyMap.get(kmer));
    }

    private static int getSuccessorCount( final int[] counts ) {
        return (counts[4] > 0 ? 1 : 0) + (counts[5] > 0 ? 1 : 0) + (counts[6] > 0 ? 1 : 0) + (counts[7] > 0 ? 1 : 0);
    }

    private static void buildContig( final SVKmerLong kmerArg, final int kSize, final int[] countsArg,
                                     final Map<SVKmerLong, int[]> kmerAdjacencyMap, final List<String> contigs ) {
        final StringBuilder sb = new StringBuilder(kmerArg.toString(kSize));
        SVKmerLong kmer = kmerArg;
        int[] counts = countsArg;
        while ( (kmer = getSoleSuccessor(kmer, kSize, counts)) != null ) {
            counts = kmerAdjacencyMap.get(kmer.canonical(kSize));
            if ( getPredecessorCount(counts) != 1 ) break;
            sb.append(kmer.lastBase().toString());
        }
        contigs.add(sb.toString());
    }
}
