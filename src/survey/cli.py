# src/survey/cli.py
import argparse
import logging
import sys
import textwrap as tw

import matplotlib as mpl

from survey.spatial import segment as spatial_segment

def segment_command(args):
    """Sub-command logic for segmentation."""

    if args.backend:
        try:
            mpl.use(args.backend)
            logging.info(f"Using Matplotlib backend: {args.backend}")
        except ImportError:
            logging.error(f"Failed to set backend '{args.backend}'. Is it installed?")
            sys.exit(1)
    
    # Parse the comma-separated string of chipnums into a list.
    # The core function now receives a clean list, not a raw string.
    parsed_chipnums = list(map(int, args.chipnums.strip().split(','))) if args.chipnums else None

    try:
        spatial_segment.run_segmentation(
            file_name=args.file_name,
            chipnums=parsed_chipnums,
            color=args.color,
            size=args.size,
            group=args.group,
            delete=args.delete,
            imgdir=args.imgdir
        )
    except (ValueError, FileNotFoundError) as e:
        # Catch expected errors and log them cleanly for the user
        logging.error(e)
        sys.exit(1)


def main():
    # --- Configure logging for the command-line interface ---
    # This setup directs log messages from the library to the console.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(
        description='A suite of survey tools.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Sub-parser for the 'segment' command ---
    segment_parser = subparsers.add_parser('segment', help='Segment spatial chips interactively.', 
                                           formatter_class=argparse.RawDescriptionHelpFormatter)

    segment_parser.description = \
        """
        Launches an interactive segmentation tool for spatial chips. The tool allows users to
        manually segment chips based on their orthogonal imaging and save the results for
        downstream analysis. The segmentation results can be used for various analyses, including
        spatial pattern recognition and cell type identification. This tool requires that a 
        svp.core.TissueImage object be present in imgs attribute of the svp.core.Chip(s).
        """

    ## --- Positional Arguments ---
    segment_parser.add_argument("file_name", type=str,
                                help="Path to an h5mu file with an `xyz` modality.")
    
    segment_parser.add_argument("imgdir", type=str,
                            help="Directory containing image files.")
    
    ## --- Optional Arguments ---
    segment_parser.add_argument("--chipnums", default=None, type=str,
                                help="Comma-separated chip numbers to process (e.g., '1,2,5'). "
                                     "If not provided, all chipnums will be processed.")
    
    segment_parser.add_argument("--color", default="leiden", type=str,
                                help="The column in `mdata.obs` to use to color the cells (default: 'leiden').")

    segment_parser.add_argument("--size", default=10, type=float,
                                help="Size of the plotted cells (default: 10).")

    segment_parser.add_argument("--group", default="tissue", type=str,
                                help="The column name for storing segmentation results (default: 'tissue').")

    segment_parser.add_argument("--delete", action="store_true",
                                help="If provided, delete any existing segmentation data in the --group column before starting.")

    segment_parser.add_argument("--backend", default=None, type=str,
                                help="Name of the Matplotlib backend to use (e.g., 'TkAgg', 'Qt5Agg')."
                                )
    
    segment_parser.set_defaults(func=segment_command)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()