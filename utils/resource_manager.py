import os
import pickle
from datetime import datetime


def ensure_directories_exist(directories):
    """Ensure necessary directories exist"""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def load_parking_positions(config_dir, reference_image):
    """Load parking positions from file"""
    try:
        pos_file = os.path.join(config_dir, f'CarParkPos_{os.path.splitext(reference_image)[0]}')

        if os.path.exists(pos_file):
            with open(pos_file, 'rb') as f:
                return pickle.load(f)
        else:
            return []

    except Exception as e:
        print(f"Error loading parking positions: {str(e)}")
        return []


def save_parking_positions(pos_list, config_dir, reference_image):
    """Save parking positions to file"""
    try:
        pos_file = os.path.join(config_dir, f'CarParkPos_{os.path.splitext(reference_image)[0]}')
        with open(pos_file, 'wb') as f:
            pickle.dump(pos_list, f)
        return True
    except Exception as e:
        print(f"Error saving parking positions: {str(e)}")
        return False


def save_log(log_data, log_dir):
    """Save log data to file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"parking_log_{timestamp}.txt")

        with open(filename, 'w') as f:
            for entry in log_data:
                f.write(entry + "\n")

        return filename
    except Exception as e:
        print(f"Error saving log: {str(e)}")
        return None


def export_statistics(stats_data, log_dir):
    """Export statistics to CSV"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"parking_stats_{timestamp}.csv")

        with open(filename, 'w') as f:
            f.write("Timestamp,Total Spaces,Free Spaces,Occupied Spaces,Vehicles Counted\n")

            for row in stats_data:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

        return filename
    except Exception as e:
        print(f"Error exporting statistics: {str(e)}")
        return None