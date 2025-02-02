# -*- coding: utf-8 -*-
import pathlib
import argparse
import json
import os
from tqdm import tqdm

from OCC.Core.TopoDS import TopoDS_Solid
from OCC.Core.STEPControl import STEPControl_Reader

from Utils.occ_utils import list_face

# occwl
from occwl.solid import Solid
from occwl.graph import face_adjacency

from topologyCheker import TopologyChecker


def load_body_from_step(step_file):
    """
    Load the body from the step file.  
    We expect only one body in each file
    """
    assert pathlib.Path(step_file).suffix in ['.step', '.stp']
    reader = STEPControl_Reader()
    reader.ReadFile(str(step_file))
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


def get_filenames(path, suffix):
    path = pathlib.Path(path)
    files = list(
        x for x in path.rglob(suffix)
    )
    files.sort()
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to load the dataset from")
    args = parser.parse_args()
    # load dataset
    aag_path = os.path.join(args.dataset, 'aag', 'graphs.json')
    step_path = os.path.join(args.dataset, 'steps')
    labels_path = os.path.join(args.dataset, 'labels')
    # AGG exists
    agg_exist = os.path.exists(aag_path)
    if agg_exist:
        print('AGG json exists')
        try:
            agg = load_json(aag_path)
        except Exception as e:
            assert False, e
    step_files = get_filenames(step_path, f"*.st*p")
    labels_files = get_filenames(labels_path, '*.json')
    # check number of files
    if agg_exist:
        assert len(agg) == len(step_files), \
            'number of AGG ({}) is not equal to number of step files ({})' .format(
            len(agg), len(step_files))
    assert len(step_files) == len(labels_files), \
        'number of label files ({}) is not equal to number of step files ({})' .format(
            len(labels_files), len(step_files))
    
    wrong_files = []
    if agg_exist:
        # loop over AAgraph, step_file, label 
        for agg_data, step_file, labels_file in tqdm(
                zip(agg, step_files, labels_files), total=len(step_files)):
            # check file name
            fn, graph = agg_data
            assert fn == step_file.stem
            assert step_file.stem == labels_file.stem
            # load step, label json
            try:
                shape = load_body_from_step(step_file)
            except Exception as e:
                print(fn, e)
                wrong_files.append((step_file, labels_file))
                continue
            try: 
                label_data = load_json(labels_file)
            except Exception as e:
                print(fn, e)
                wrong_files.append((step_file, labels_file))
                continue
            # crosscheck labels
            faces_list = list_face(shape)
            num_faces = len(faces_list)
            # check length of aag equals to number of faces
            if num_faces != graph['graph']['num_nodes']:
                print('File {} have wrong number of labels {} with AAG faces {}. '.format(
                    fn, num_faces, graph['graph']['num_nodes']))
                wrong_files.append((step_file, labels_file))
                continue
            # check length of label
            file_id, label = label_data[0]
            seg_label, inst_label, bottom_label = label['seg'], label['inst'], label['bottom']
            # check map between face id and segmentation label
            if num_faces != len(seg_label):
                print('File {} have wrong number of seg labels {} with step faces {}. '.format(
                    fn, len(seg_label), num_faces))
                wrong_files.append((step_file, labels_file))
                continue
            # check relation_matrix describing the feature instances
            if num_faces != len(inst_label):
                print('File {} have wrong number of instance labels {} with step faces {}. '.format(
                    fn, len(inst_label), num_faces))
                wrong_files.append((step_file, labels_file))
                continue
            # check map between face id and bottom identification label
            if num_faces != len(bottom_label):
                print('File {} have wrong number of bottom labels {} with step faces {}. '.format(
                    fn, len(bottom_label), num_faces))
                wrong_files.append((step_file, labels_file))
                continue
    else:
        # loop over step_file, label
        topochecker = TopologyChecker()
        for step_file, labels_file in tqdm(
                zip(step_files, labels_files), total=len(step_files)):
            # check file name
            fn = step_file.stem
            assert fn == labels_file.stem, f'{step_file.stem}, {labels_file.stem}'
            # load step, label json
            try:
                shape = load_body_from_step(step_file)
            except Exception as e:
                print('file', fn)
                print(e)
                wrong_files.append((step_file, labels_file))
                continue
            try: 
                label_data = load_json(labels_file)
            except Exception as e:
                print('file', fn)
                print(e)
                wrong_files.append((step_file, labels_file))
                continue
            # check shape is TopoDS_Solid
            if not isinstance(shape, TopoDS_Solid):
                print('{} is {}, not supported'.format(fn, type(shape)))
                wrong_files.append((step_file, labels_file))
                continue
            # check shape topology
            if not topochecker(shape):
                print("{} has wrong topology".format(fn))
                wrong_files.append((step_file, labels_file))
                continue
            # check generated shape can be exported to face_adjacency
            try:
                graph = face_adjacency(Solid(shape))
            except Exception as e:
                print('Wrong {} with Exception:'.format(fn))
                print(e)
                wrong_files.append((step_file, labels_file))
                continue
            # crosscheck labels
            faces_list = list_face(shape)
            num_faces = len(faces_list)
            # check length of label
            file_id, label = label_data[0]
            seg_label, inst_label, bottom_label = label['seg'], label['inst'], label['bottom']
            # check map between face id and segmentation label
            if num_faces != len(seg_label):
                print('File {} have wrong number of seg labels {} with step faces {}. '.format(
                    fn, len(seg_label), num_faces))
                wrong_files.append((step_file, labels_file))
                continue
            # check relation_matrix describing the feature instances
            if num_faces != len(inst_label):
                print('File {} have wrong number of instance labels {} with step faces {}. '.format(
                    fn, len(inst_label), num_faces))
                wrong_files.append((step_file, labels_file))
                continue
            # check map between face id and bottom identification label
            if num_faces != len(bottom_label):
                print('File {} have wrong number of bottom labels {} with step faces {}. '.format(
                    fn, len(bottom_label), num_faces))
                wrong_files.append((step_file, labels_file))
                continue
    # delete wrong steps and labels or not
    if len(wrong_files):
        print('delete following wrong files:')
        print(wrong_files)
        inputs = input('[Y/N]: ')
        if (inputs == 'Y') or (inputs == 'y'):
            for wrong_file in wrong_files:
                step_f, label_f = wrong_file
                os.remove(step_f)
                os.remove(label_f)
                print(step_f, label_f, 'deleted')
            if agg_exist:
                os.remove(aag_path)
                print(aag_path, 'deleted, please regenerate AAG')

    print('Finished')
